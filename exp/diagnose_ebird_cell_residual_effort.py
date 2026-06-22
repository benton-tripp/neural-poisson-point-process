"""
Diagnose whether spatial GNN residual corrections track effort/access geography.

This is a framework diagnostic. It aggregates held-out checklists by spatial GNN
cell and compares the model's spatial residual probability correction against
effort, access, environmental, and observed-prevalence summaries. Strong
correlation with checklist density or protocol/effort variables is a warning
that the spatial correction may be absorbing citizen-science observation bias
rather than only ecological suitability.

Run from the project root:

    python exp/diagnose_ebird_cell_residual_effort.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_residual_scaled_sigmoid010_l2_0p01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from diagnose_ebird_block_species import (
    assign_spatial_blocks,
    load_spatial_run,
    load_spatial_probabilities,
)
from ebird_graph_all_species_baseline import build_label_matrix, load_split_checklists
from ebird_joint_tabular_baseline import SEED
from ebird_spatial_gnn_baseline import (
    SpatialGCNHybrid,
    build_cell_species_support_features,
    build_species_adjacency_for_run,
    build_spatial_cell_graph,
    build_spatial_cell_graph_for_run,
    inject_frozen_access_embeddings,
    load_frozen_access_embeddings,
)


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial_10x10"
DEFAULT_SPATIAL_RUN = "spatial_gcn_residual_scaled_sigmoid010_l2_0p01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate held-out spatial residual corrections by cell and compare "
            "them with effort/access/ecology summaries."
        )
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
        help=(
            "Output directory. Defaults to "
            "graph-dir/spatial_gnn_baselines/diagnostics/cell_residual_effort."
        ),
    )
    parser.add_argument(
        "--min-cell-checklists",
        type=int,
        default=25,
        help="Minimum held-out checklists required for cell-level summaries. Defaults to 25.",
    )
    parser.add_argument(
        "--top-cells",
        type=int,
        default=20,
        help="Rows to print for largest residual cells. Defaults to 20.",
    )
    parser.add_argument(
        "--gnn-batch-size",
        type=int,
        default=2048,
        help="Spatial GNN evaluation batch size. Defaults to 2048.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed. Defaults to 19.")
    return parser.parse_args()


def safe_corr(frame: pd.DataFrame, x: str, y: str, method: str) -> float:
    values = frame[[x, y]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) < 3:
        return np.nan
    if values[x].nunique() < 2 or values[y].nunique() < 2:
        return np.nan
    return float(values[x].corr(values[y], method=method))


def summarize_cells(
    nodes: pd.DataFrame,
    labels: np.ndarray,
    base_scores: np.ndarray,
    full_scores: np.ndarray,
    checklist_cells: np.ndarray,
    spatial_blocks: np.ndarray,
    min_cell_checklists: int,
    access_bias_logits: np.ndarray | None = None,
    access_delta_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    delta = full_scores - base_scores
    labels_bool = labels.astype(bool)
    rows = []
    for cell_id in sorted(pd.unique(checklist_cells)):
        mask = checklist_cells == cell_id
        count = int(mask.sum())
        if count < min_cell_checklists:
            continue
        cell_nodes = nodes.loc[mask]
        cell_labels = labels[mask]
        cell_labels_bool = labels_bool[mask]
        cell_delta = delta[mask]
        cell_access_logits = access_bias_logits[mask] if access_bias_logits is not None else None
        cell_access_delta = access_delta_scores[mask] if access_delta_scores is not None else None
        positives = int(cell_labels.sum())
        negatives = int(cell_labels.size - positives)
        positive_deltas = cell_delta[cell_labels_bool]
        negative_deltas = cell_delta[~cell_labels_bool]
        row = {
            "spatial_cell": int(cell_id),
            "spatial_block": int(pd.Series(spatial_blocks[mask]).mode().iloc[0]),
            "checklists": count,
            "x_mean": float(cell_nodes["x"].mean()),
            "y_mean": float(cell_nodes["y"].mean()),
            "observed_rate": float(cell_labels.mean()),
            "species_per_checklist_mean": float(cell_labels.sum(axis=1).mean()),
            "species_per_checklist_p90": float(np.quantile(cell_labels.sum(axis=1), 0.9)),
            "positive_pairs": positives,
            "negative_pairs": negatives,
            "base_probability_mean": float(base_scores[mask].mean()),
            "full_probability_mean": float(full_scores[mask].mean()),
            "probability_delta_mean": float(cell_delta.mean()),
            "probability_delta_abs_mean": float(np.abs(cell_delta).mean()),
            "probability_delta_p10": float(np.quantile(cell_delta, 0.10)),
            "probability_delta_p90": float(np.quantile(cell_delta, 0.90)),
            "positive_probability_delta_mean": float(positive_deltas.mean())
            if len(positive_deltas)
            else np.nan,
            "negative_probability_delta_mean": float(negative_deltas.mean())
            if len(negative_deltas)
            else np.nan,
            "duration_minutes_mean": float(cell_nodes["duration_minutes"].mean()),
            "duration_minutes_p90": float(cell_nodes["duration_minutes"].quantile(0.90)),
            "effort_distance_km_mean": float(cell_nodes["effort_distance_km"].mean()),
            "effort_distance_km_p90": float(cell_nodes["effort_distance_km"].quantile(0.90)),
            "number_observers_mean": float(cell_nodes["number_observers"].mean()),
            "traveling_rate": float((cell_nodes["protocol_name"] == "Traveling").mean()),
            "stationary_rate": float((cell_nodes["protocol_name"] == "Stationary").mean()),
            "unique_observers": int(cell_nodes["observer_id"].nunique()),
            "unique_localities": int(cell_nodes["locality_id"].nunique()),
            "canopy_median_mean": float(cell_nodes["canopy_median"].mean()),
            "elevation_mean": float(cell_nodes["nc_usgs30m_match_tcc"].mean()),
            "distance_to_waterbody_m_mean": float(
                cell_nodes["distance_to_waterbody_m"].mean()
            ),
            "distance_to_coastline_m_mean": float(
                cell_nodes["distance_to_coastline_m"].mean()
            ),
        }
        if cell_access_logits is not None:
            row["access_bias_logit_mean"] = float(cell_access_logits.mean())
            row["access_bias_logit_abs_mean"] = float(np.abs(cell_access_logits).mean())
            row["access_bias_logit_p10"] = float(np.quantile(cell_access_logits, 0.10))
            row["access_bias_logit_p90"] = float(np.quantile(cell_access_logits, 0.90))
        if cell_access_delta is not None:
            row["access_probability_delta_mean"] = float(cell_access_delta.mean())
            row["access_probability_delta_abs_mean"] = float(np.abs(cell_access_delta).mean())
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["log_checklists"] = np.log1p(frame["checklists"])
        frame["observer_per_checklist"] = frame["unique_observers"] / frame["checklists"]
        frame["locality_per_checklist"] = frame["unique_localities"] / frame["checklists"]
    return frame


def correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    predictors = [
        "log_checklists",
        "observed_rate",
        "species_per_checklist_mean",
        "duration_minutes_mean",
        "duration_minutes_p90",
        "effort_distance_km_mean",
        "effort_distance_km_p90",
        "number_observers_mean",
        "traveling_rate",
        "stationary_rate",
        "observer_per_checklist",
        "locality_per_checklist",
        "canopy_median_mean",
        "elevation_mean",
        "distance_to_waterbody_m_mean",
        "distance_to_coastline_m_mean",
    ]
    targets = [
        "probability_delta_mean",
        "probability_delta_abs_mean",
        "positive_probability_delta_mean",
        "negative_probability_delta_mean",
        "full_probability_mean",
        "access_bias_logit_mean",
        "access_bias_logit_abs_mean",
        "access_probability_delta_mean",
        "access_probability_delta_abs_mean",
    ]
    rows = []
    for target in targets:
        for predictor in predictors:
            if predictor not in frame.columns or target not in frame.columns:
                continue
            rows.append(
                {
                    "target": target,
                    "predictor": predictor,
                    "pearson": safe_corr(frame, predictor, target, "pearson"),
                    "spearman": safe_corr(frame, predictor, target, "spearman"),
                    "cells": int(frame[[predictor, target]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target", "spearman"], key=lambda col: col.abs() if col.name == "spearman" else col
    )


def load_spatial_components(
    graph_dir: Path,
    run_name: str,
    features_np: np.ndarray,
    train_checklists: np.ndarray,
    test_checklists: np.ndarray,
    species_count: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
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
    access_bias_parts = []
    access_delta_parts = []
    with torch.no_grad():
        cell_embeddings = model.encode_cells(cell_features, adjacency)
        cell_embeddings = inject_frozen_access_embeddings(
            model, cell_embeddings, frozen_access_embeddings
        )
        for start in range(0, len(test_checklists), batch_size):
            batch = test_checklists[start : start + batch_size]
            batch_tensor = torch.from_numpy(batch.astype(np.int64))
            base_logits, full_logits = model.base_and_full_logits(
                features[batch_tensor],
                cell_embeddings,
                checklist_cells[batch_tensor],
                species_adjacency,
                cell_features=cell_features,
                cell_species_support_features=cell_species_support_features,
            )
            base_scores = torch.sigmoid(base_logits)
            base_parts.append(base_scores.cpu().numpy())
            full_parts.append(torch.sigmoid(full_logits).cpu().numpy())
            if model.last_spatial_access_bias_logits is not None:
                access_logits = model.last_spatial_access_bias_logits
                no_access_scores = torch.sigmoid(base_logits - access_logits)
                access_bias_parts.append(access_logits.cpu().numpy())
                access_delta_parts.append((base_scores - no_access_scores).cpu().numpy())
    access_bias = (
        np.vstack(access_bias_parts).astype(np.float32) if access_bias_parts else None
    )
    access_delta = (
        np.vstack(access_delta_parts).astype(np.float32) if access_delta_parts else None
    )
    return (
        np.vstack(base_parts).astype(np.float32),
        np.vstack(full_parts).astype(np.float32),
        access_bias,
        access_delta,
        summary,
    )


def plot_scatter_grid(frame: pd.DataFrame, output_dir: Path) -> None:
    if frame.empty:
        return
    pairs = [
        ("log_checklists", "probability_delta_abs_mean", "log checklist count", "mean |residual delta|"),
        ("traveling_rate", "probability_delta_mean", "traveling rate", "mean residual delta"),
        ("duration_minutes_mean", "probability_delta_mean", "mean duration", "mean residual delta"),
        ("observed_rate", "probability_delta_mean", "observed pair rate", "mean residual delta"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (x, y, xlabel, ylabel) in zip(axes.ravel(), pairs):
        ax.scatter(frame[x], frame[y], s=np.clip(frame["checklists"] / 20, 8, 80), alpha=0.65)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        rho = safe_corr(frame, x, y, "spearman")
        ax.set_title(f"Spearman rho={rho:.3f}" if np.isfinite(rho) else "Spearman rho=NA")
    plt.tight_layout()
    plt.savefig(output_dir / "cell_residual_effort_scatter.png", dpi=180)
    plt.close(fig)


def plot_cell_maps(frame: pd.DataFrame, output_dir: Path) -> None:
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plots = [
        ("probability_delta_mean", "Mean residual probability delta", "coolwarm"),
        ("probability_delta_abs_mean", "Mean |residual probability delta|", "magma"),
        ("log_checklists", "Held-out checklist density", "viridis"),
    ]
    for ax, (column, title, cmap) in zip(axes, plots):
        values = frame[column]
        vlim = None
        if column == "probability_delta_mean":
            vlim = float(np.nanquantile(np.abs(values), 0.98))
        scatter = ax.scatter(
            frame["x_mean"],
            frame["y_mean"],
            c=values,
            s=np.clip(frame["checklists"] / 25, 8, 100),
            cmap=cmap,
            alpha=0.85,
            vmin=-vlim if vlim else None,
            vmax=vlim if vlim else None,
            linewidths=0,
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(scatter, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_dir / "cell_residual_effort_maps.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else graph_dir / "spatial_gnn_baselines" / "diagnostics" / "cell_residual_effort"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    species_count = int(metadata["counts"]["species"])
    features = np.load(graph_dir / "checklist_features.npy").astype(np.float32)
    train_checklists = load_split_checklists(graph_dir, "train", None, args.seed)
    test_checklists = load_split_checklists(graph_dir, "test", None, args.seed)
    labels = build_label_matrix(graph_dir, test_checklists, species_count, "test")
    base_scores, full_scores, access_bias_logits, access_delta_scores, spatial_summary = load_spatial_components(
        graph_dir,
        args.spatial_run_name,
        features,
        train_checklists,
        test_checklists,
        species_count,
        args.gnn_batch_size,
    )

    model_info = spatial_summary["model"]
    _cell_features, _adjacency, checklist_cell, cell_metadata = build_spatial_cell_graph(
        graph_dir,
        features,
        train_checklists,
        float(model_info["spatial_grid_size_m"]),
    )
    nodes = pd.read_parquet(graph_dir / "checklist_nodes.parquet")
    nodes = nodes.set_index("checklist_index").loc[test_checklists].reset_index()
    test_cells = checklist_cell[test_checklists]
    blocks_per_dim = int(metadata.get("split", {}).get("spatial_blocks_per_dim", 8))
    all_block_nodes = pd.read_parquet(
        graph_dir / "checklist_nodes.parquet", columns=["checklist_index", "x", "y"]
    )
    all_blocks = assign_spatial_blocks(all_block_nodes, blocks_per_dim)
    test_blocks = all_blocks[test_checklists]

    cell_summary = summarize_cells(
        nodes,
        labels,
        base_scores,
        full_scores,
        test_cells,
        test_blocks,
        args.min_cell_checklists,
        access_bias_logits=access_bias_logits,
        access_delta_scores=access_delta_scores,
    )
    correlations = correlation_table(cell_summary)
    cell_summary.to_csv(output_dir / f"{args.spatial_run_name}_cell_summary.csv", index=False)
    correlations.to_csv(
        output_dir / f"{args.spatial_run_name}_cell_correlations.csv", index=False
    )
    plot_scatter_grid(cell_summary, output_dir)
    plot_cell_maps(cell_summary, output_dir)
    (output_dir / f"{args.spatial_run_name}_metadata.json").write_text(
        json.dumps(
            {
                "graph_dir": str(graph_dir),
                "spatial_run_name": args.spatial_run_name,
                "min_cell_checklists": args.min_cell_checklists,
                "test_checklists": int(len(test_checklists)),
                "reported_cells": int(len(cell_summary)),
                "cell_metadata": cell_metadata,
                "spatial_summary": spatial_summary,
                "outputs": {
                    "cell_summary": f"{args.spatial_run_name}_cell_summary.csv",
                    "cell_correlations": f"{args.spatial_run_name}_cell_correlations.csv",
                    "scatter": "cell_residual_effort_scatter.png",
                    "maps": "cell_residual_effort_maps.png",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote cell residual/effort diagnostics to {output_dir}")
    if cell_summary.empty:
        print("No cells met the minimum checklist threshold.")
        return
    print(f"Cells summarized: {len(cell_summary):,}")
    print("\nLargest mean absolute residual probability deltas:")
    display_columns = [
        "spatial_cell",
        "spatial_block",
        "checklists",
        "observed_rate",
        "probability_delta_mean",
        "probability_delta_abs_mean",
        "traveling_rate",
        "duration_minutes_mean",
        "effort_distance_km_mean",
        "unique_observers",
        "unique_localities",
    ]
    print(
        cell_summary.sort_values("probability_delta_abs_mean", ascending=False)
        [display_columns]
        .head(args.top_cells)
        .to_string(index=False, float_format="%.4f")
    )
    print("\nStrongest Spearman correlations with residual summaries:")
    print(
        correlations.reindex(correlations["spearman"].abs().sort_values(ascending=False).index)
        .head(args.top_cells)
        .to_string(index=False, float_format="%.4f")
    )


if __name__ == "__main__":
    main()
