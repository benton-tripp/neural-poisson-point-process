"""
Compare tabular MLP and spatial GNN performance within effort/spatial strata.

This retrains the tabular MLP on the graph dataset's standardized checklist
features, then evaluates it and a saved spatial GNN on the same held-out
all-pairs species/checklist target within protocol, duration, distance,
observer-count, and spatial-block strata.

Run from the project root:

    python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial --spatial-run-name spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001
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
    evaluate_all_pairs,
    inject_frozen_access_embeddings,
    load_frozen_access_embeddings,
)


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial"
DEFAULT_SPATIAL_RUN = "spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare tabular MLP and spatial GNN metrics by effort strata."
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
        help="Output directory. Defaults to graph-dir/spatial_gnn_baselines/diagnostics/effort_strata.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Tabular MLP epochs. Defaults to 50.")
    parser.add_argument("--batch-size", type=int, default=8192, help="Tabular MLP batch size. Defaults to 8192.")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="Tabular MLP learning rate. Defaults to 1e-2.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Tabular MLP weight decay. Defaults to 1e-4.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Tabular MLP hidden width. Defaults to 64.")
    parser.add_argument("--hidden-layers", type=int, default=1, help="Tabular MLP hidden layers. Defaults to 1.")
    parser.add_argument("--dropout", type=float, default=0.10, help="Tabular MLP dropout. Defaults to 0.10.")
    parser.add_argument("--gnn-batch-size", type=int, default=2048, help="Spatial GNN evaluation batch size. Defaults to 2048.")
    parser.add_argument("--calibration-bins", type=int, default=10, help="Calibration bins. Defaults to 10.")
    parser.add_argument("--min-checklists", type=int, default=1000, help="Minimum checklists for stratum reporting. Defaults to 1000.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed. Defaults to 19.")
    return parser.parse_args()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


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


def load_spatial_scores(
    graph_dir: Path,
    run_name: str,
    features_np: np.ndarray,
    train_checklists: np.ndarray,
    test_checklists: np.ndarray,
    labels_test: np.ndarray,
    species: pd.DataFrame,
    batch_size: int,
    calibration_bins: int,
) -> tuple[np.ndarray, dict]:
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
        graph_dir, train_checklists, len(species), "train"
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
        species_count=len(species),
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
    score_summary, _species_metrics, _calibration = evaluate_all_pairs(
        model,
        features,
        cell_features,
        cell_species_support_features,
        adjacency,
        frozen_access_embeddings,
        species_adjacency,
        torch.from_numpy(checklist_cell.astype(np.int64)),
        test_checklists,
        labels_test,
        species,
        batch_size,
        calibration_bins,
    )

    score_parts = []
    with torch.no_grad():
        cell_embeddings = model.encode_cells(cell_features, adjacency)
        cell_embeddings = inject_frozen_access_embeddings(
            model, cell_embeddings, frozen_access_embeddings
        )
        for start in range(0, len(test_checklists), batch_size):
            batch = test_checklists[start : start + batch_size]
            batch_tensor = torch.from_numpy(batch.astype(np.int64))
            logits = model.forward_all_species(
                features[batch_tensor],
                cell_embeddings,
                torch.from_numpy(checklist_cell[batch].astype(np.int64)),
                species_adjacency,
                cell_features=cell_features,
                cell_species_support_features=cell_species_support_features,
            )
            score_parts.append(torch.sigmoid(logits).cpu().numpy())
    return np.vstack(score_parts).astype(np.float32), score_summary


def assign_spatial_blocks(nodes: pd.DataFrame, blocks_per_dim: int) -> np.ndarray:
    x = nodes["x"].to_numpy(dtype=np.float64)
    y = nodes["y"].to_numpy(dtype=np.float64)
    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    x_bin = np.floor((x - x.min()) / x_span * blocks_per_dim).astype(np.int64)
    y_bin = np.floor((y - y.min()) / y_span * blocks_per_dim).astype(np.int64)
    x_bin = np.clip(x_bin, 0, blocks_per_dim - 1)
    y_bin = np.clip(y_bin, 0, blocks_per_dim - 1)
    return y_bin * blocks_per_dim + x_bin


def build_strata(nodes: pd.DataFrame) -> pd.DataFrame:
    strata = pd.DataFrame(index=nodes.index)
    strata["protocol"] = nodes["protocol_name"].astype(str)
    strata["duration"] = pd.cut(
        nodes["duration_minutes"].astype(float),
        bins=[0, 10, 30, 60, 120, np.inf],
        labels=["1-10", "11-30", "31-60", "61-120", "121+"],
        include_lowest=True,
    ).astype(str)
    strata["distance"] = pd.cut(
        nodes["effort_distance_km"].fillna(0.0).astype(float),
        bins=[-0.001, 0, 0.5, 2, 5, np.inf],
        labels=["0", "(0,0.5]", "(0.5,2]", "(2,5]", "5+"],
        include_lowest=True,
    ).astype(str)
    strata["observers"] = pd.cut(
        nodes["number_observers"].astype(float),
        bins=[0, 1, 2, np.inf],
        labels=["1", "2", "3+"],
        include_lowest=True,
    ).astype(str)
    strata["spatial_block"] = nodes["spatial_block"].astype(str)
    return strata


def calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int) -> tuple[float, float]:
    labels_flat = labels.reshape(-1)
    scores_flat = scores.reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    max_error = 0.0
    total = len(scores_flat)
    for idx in range(bins):
        if idx == bins - 1:
            mask = (scores_flat >= edges[idx]) & (scores_flat <= edges[idx + 1])
        else:
            mask = (scores_flat >= edges[idx]) & (scores_flat < edges[idx + 1])
        if not mask.any():
            continue
        error = abs(float(scores_flat[mask].mean()) - float(labels_flat[mask].mean()))
        ece += error * float(mask.sum()) / total
        max_error = max(max_error, error)
    return float(ece), float(max_error)


def metric_row(
    model_name: str,
    stratum_type: str,
    stratum: str,
    labels: np.ndarray,
    scores: np.ndarray,
    calibration_bins: int,
) -> dict:
    labels_flat = labels.reshape(-1)
    scores_flat = scores.reshape(-1)
    ece, max_error = calibration_error(labels, scores, calibration_bins)
    species_auroc = []
    species_auprc = []
    for species_idx in range(labels.shape[1]):
        species_auroc.append(auc_roc(labels[:, species_idx], scores[:, species_idx]))
        species_auprc.append(average_precision(labels[:, species_idx], scores[:, species_idx]))
    return {
        "model": model_name,
        "stratum_type": stratum_type,
        "stratum": stratum,
        "checklists": int(labels.shape[0]),
        "pairs": int(labels_flat.size),
        "positives": int(labels_flat.sum()),
        "observed_rate": float(labels_flat.mean()),
        "mean_predicted": float(scores_flat.mean()),
        "micro_auroc": auc_roc(labels_flat, scores_flat),
        "micro_auprc": average_precision(labels_flat, scores_flat),
        "macro_auroc": float(np.nanmean(species_auroc)),
        "macro_auprc": float(np.nanmean(species_auprc)),
        "ece": ece,
        "max_bin_error": max_error,
        "calibration_error": float(abs(scores_flat.mean() - labels_flat.mean())),
    }


def compare_by_strata(
    labels: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    strata: pd.DataFrame,
    min_checklists: int,
    calibration_bins: int,
) -> pd.DataFrame:
    rows = []
    for stratum_type in strata.columns:
        for stratum, index in strata.groupby(stratum_type, sort=True).groups.items():
            positions = np.array(list(index), dtype=np.int64)
            if len(positions) < min_checklists:
                continue
            for model_name, scores in scores_by_model.items():
                rows.append(
                    metric_row(
                        model_name,
                        stratum_type,
                        str(stratum),
                        labels[positions],
                        scores[positions],
                        calibration_bins,
                    )
                )
    metrics = pd.DataFrame(rows)
    if metrics.empty:
        return metrics
    wide = metrics.pivot_table(
        index=["stratum_type", "stratum"],
        columns="model",
        values=["micro_auprc", "macro_auprc", "micro_auroc", "macro_auroc", "ece", "calibration_error"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    for metric in ["micro_auprc", "macro_auprc", "micro_auroc", "macro_auroc", "ece", "calibration_error"]:
        left = f"{metric}_spatial_gnn"
        right = f"{metric}_tabular_mlp"
        if left in wide.columns and right in wide.columns:
            wide[f"delta_{metric}"] = wide[left] - wide[right]
    return metrics.merge(wide, on=["stratum_type", "stratum"], how="left")


def plot_strata(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    deltas = (
        metrics[["stratum_type", "stratum", "delta_micro_auprc", "delta_ece"]]
        .drop_duplicates()
        .copy()
    )
    for stratum_type, frame in deltas.groupby("stratum_type"):
        frame = frame.sort_values("delta_micro_auprc")
        fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.45 * len(frame))))
        axes[0].barh(frame["stratum"], frame["delta_micro_auprc"], color="#4C78A8")
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_title("Delta micro AUPRC")
        axes[0].set_xlabel("spatial GNN - tabular MLP")
        axes[1].barh(frame["stratum"], frame["delta_ece"], color="#F58518")
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_title("Delta ECE")
        axes[1].set_xlabel("spatial GNN - tabular MLP")
        fig.suptitle(stratum_type)
        plt.tight_layout()
        safe = "".join(ch if ch.isalnum() else "_" for ch in stratum_type.lower()).strip("_")
        plt.savefig(output_dir / f"{safe}_strata_deltas.png", dpi=180)
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
        else graph_dir / "spatial_gnn_baselines" / "diagnostics" / "effort_strata"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir = output_dir / safe_name(args.spatial_run_name)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    species = pd.read_csv(graph_dir / "species_nodes.csv")
    features = np.load(graph_dir / "checklist_features.npy").astype(np.float32)
    train_checklists = load_split_checklists(graph_dir, "train", None, args.seed)
    test_checklists = load_split_checklists(graph_dir, "test", None, args.seed)
    y_train = build_label_matrix(graph_dir, train_checklists, len(species), "train")
    y_test = build_label_matrix(graph_dir, test_checklists, len(species), "test")

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
    spatial_scores, spatial_summary = load_spatial_scores(
        graph_dir,
        args.spatial_run_name,
        features,
        train_checklists,
        test_checklists,
        y_test,
        species,
        args.gnn_batch_size,
        args.calibration_bins,
    )

    node_columns = [
        "checklist_index",
        "protocol_name",
        "duration_minutes",
        "effort_distance_km",
        "number_observers",
        "x",
        "y",
    ]
    all_nodes = pd.read_parquet(graph_dir / "checklist_nodes.parquet", columns=node_columns)
    blocks_per_dim = int(metadata.get("split", {}).get("spatial_blocks_per_dim", 8))
    all_nodes["spatial_block"] = assign_spatial_blocks(all_nodes, blocks_per_dim)
    nodes = all_nodes.set_index("checklist_index").loc[test_checklists].reset_index()
    strata = build_strata(nodes)
    metrics = compare_by_strata(
        y_test,
        {"tabular_mlp": tabular_scores, "spatial_gnn": spatial_scores},
        strata,
        args.min_checklists,
        args.calibration_bins,
    )
    metadata_payload = {
        "graph_dir": str(graph_dir),
        "spatial_run_name": args.spatial_run_name,
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
        metrics.to_csv(target_dir / "effort_strata_metrics.csv", index=False)
        plot_strata(metrics, target_dir)
        (target_dir / "effort_strata_metadata.json").write_text(
            json.dumps(
                metadata_payload,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(f"Wrote effort-strata diagnostics to {run_output_dir}")
    print(f"Updated compatibility copy at {output_dir}")
    print("\nLargest spatial GNN micro-AUPRC gains over tabular:")
    print(
        metrics.drop_duplicates(["stratum_type", "stratum"])
        .sort_values("delta_micro_auprc", ascending=False)
        [
            [
                "stratum_type",
                "stratum",
                "checklists",
                "observed_rate",
                "delta_micro_auprc",
                "delta_macro_auprc",
                "delta_ece",
            ]
        ]
        .head(12)
        .to_string(index=False, float_format="%.4f")
    )
    print("\nLargest spatial GNN micro-AUPRC losses vs tabular:")
    print(
        metrics.drop_duplicates(["stratum_type", "stratum"])
        .sort_values("delta_micro_auprc")
        [
            [
                "stratum_type",
                "stratum",
                "checklists",
                "observed_rate",
                "delta_micro_auprc",
                "delta_macro_auprc",
                "delta_ece",
            ]
        ]
        .head(12)
        .to_string(index=False, float_format="%.4f")
    )


if __name__ == "__main__":
    main()
