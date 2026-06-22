"""
Summarize spatial GNN residual behavior by held-out block and species.

This diagnostic focuses on the GNN's own spatial correction:

    full prediction - base checklist/species prediction

It is useful when a species is consistently hurt by the spatial GNN and we need
to see whether one held-out block is driving the suppression or ranking issue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ebird_graph_all_species_baseline import build_label_matrix, load_split_checklists
from ebird_joint_tabular_baseline import SEED, auc_roc, average_precision
from ebird_spatial_gnn_baseline import (
    SpatialGCNHybrid,
    build_cell_species_support_features,
    build_species_adjacency_for_run,
    build_spatial_cell_graph_for_run,
    inject_frozen_access_embeddings,
    load_frozen_access_embeddings,
)
from plot_ebird_spatial_gnn_residual_maps import (
    load_positive_labels,
    load_run,
    predict_selected_species,
)


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial_10x10_coastalstress"
DEFAULT_RUN_NAME = "spatial_gcn_frozen_access_h64_l2_z64"
DEFAULT_SPECIES = [
    "Red-headed Woodpecker",
    "Northern Rough-winged Swallow",
    "Green Heron",
    "Dark-eyed Junco",
    "Scarlet Tanager",
    "Swamp Sparrow",
    "Hairy Woodpecker",
    "Common Grackle",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize residual probability deltas by block and species."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help=f"Spatial GNN run name without spatial_gnn_ prefix. Defaults to {DEFAULT_RUN_NAME}.",
    )
    parser.add_argument(
        "--species",
        nargs="*",
        default=DEFAULT_SPECIES,
        help="Species common names to summarize.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            "graph-dir/spatial_gnn_baselines/diagnostics/species_block_residuals/run-name."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Prediction batch size. Defaults to 8,192.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed. Defaults to 19.",
    )
    return parser.parse_args()


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


def build_model(
    graph_dir: Path,
    run_name: str,
    features_np: np.ndarray,
    train_checklists: np.ndarray,
    species_count: int,
) -> tuple[
    SpatialGCNHybrid,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    np.ndarray,
    dict,
    dict,
]:
    summary, model_path = load_run(graph_dir, run_name)
    model_info = summary["model"]
    cell_features, adjacency, checklist_cell, cell_metadata = build_spatial_cell_graph_for_run(
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
    species_adjacency, species_graph_metadata = build_species_adjacency_for_run(
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
    return (
        model,
        cell_features,
        adjacency,
        frozen_access_embeddings,
        species_adjacency,
        cell_species_support_features,
        checklist_cell,
        cell_metadata,
        species_graph_metadata,
    )


def probability_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    observed_rate = float(labels.mean()) if len(labels) else np.nan
    mean_predicted = float(scores.mean()) if len(scores) else np.nan
    return {
        "observed_rate": observed_rate,
        "mean_predicted": mean_predicted,
        "calibration_error": float(abs(mean_predicted - observed_rate))
        if np.isfinite(mean_predicted) and np.isfinite(observed_rate)
        else np.nan,
        "auroc": auc_roc(labels, scores) if positives > 0 and negatives > 0 else np.nan,
        "auprc": average_precision(labels, scores) if positives > 0 else np.nan,
    }


def summarize_group(
    common_name: str,
    block_id: int,
    base_prob: np.ndarray,
    full_prob: np.ndarray,
    access_delta: np.ndarray | None,
    labels: np.ndarray,
) -> dict:
    delta = full_prob - base_prob
    row = {
        "common_name": common_name,
        "spatial_block": int(block_id),
        "test_checklists": int(len(labels)),
        "positive_checklists": int(labels.sum()),
        "negative_checklists": int((~labels).sum()),
        "mean_base_probability": float(base_prob.mean()),
        "mean_full_probability": float(full_prob.mean()),
        "mean_probability_delta": float(delta.mean()),
        "mean_abs_probability_delta": float(np.abs(delta).mean()),
        "positive_mean_delta": float(delta[labels].mean()) if labels.any() else np.nan,
        "negative_mean_delta": float(delta[~labels].mean()) if (~labels).any() else np.nan,
        "positive_p90_abs_delta": float(np.percentile(np.abs(delta[labels]), 90))
        if labels.any()
        else np.nan,
        "all_p90_abs_delta": float(np.percentile(np.abs(delta), 90)),
    }
    if access_delta is not None:
        row.update(
            {
                "mean_access_probability_delta": float(access_delta.mean()),
                "mean_abs_access_probability_delta": float(np.abs(access_delta).mean()),
                "positive_mean_access_delta": float(access_delta[labels].mean())
                if labels.any()
                else np.nan,
                "negative_mean_access_delta": float(access_delta[~labels].mean())
                if (~labels).any()
                else np.nan,
            }
        )
    base_metrics = probability_metrics(labels, base_prob)
    full_metrics = probability_metrics(labels, full_prob)
    for key, value in base_metrics.items():
        row[f"base_{key}"] = value
    for key, value in full_metrics.items():
        row[f"full_{key}"] = value
    row["full_minus_base_auroc"] = row["full_auroc"] - row["base_auroc"]
    row["full_minus_base_auprc"] = row["full_auprc"] - row["base_auprc"]
    row["full_minus_base_calibration_error"] = (
        row["full_calibration_error"] - row["base_calibration_error"]
    )
    return row


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else graph_dir
        / "spatial_gnn_baselines"
        / "diagnostics"
        / "species_block_residuals"
        / args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    blocks_per_dim = int(metadata["split"]["spatial_blocks_per_dim"])
    features_np = np.load(graph_dir / "checklist_features.npy").astype(np.float32)
    features = torch.from_numpy(features_np)
    train_checklists = load_split_checklists(graph_dir, "train", None, args.seed)
    test_checklists = load_split_checklists(graph_dir, "test", None, args.seed)
    species = pd.read_csv(graph_dir / "species_nodes.csv")
    name_to_index = dict(zip(species["common_name"], species["species_index"]))

    species_indices = []
    species_names = []
    for common_name in args.species:
        if common_name not in name_to_index:
            print(f"Skipping unknown species: {common_name}")
            continue
        species_names.append(common_name)
        species_indices.append(int(name_to_index[common_name]))
    if not species_indices:
        raise ValueError("No valid species selected.")

    (
        model,
        cell_features,
        adjacency,
        frozen_access_embeddings,
        species_adjacency,
        cell_species_support_features,
        checklist_cell,
        cell_metadata,
        species_graph_metadata,
    ) = build_model(
        graph_dir,
        args.run_name,
        features_np,
        train_checklists,
        len(species),
    )

    base_prob, full_prob, access_delta = predict_selected_species(
        model,
        features,
        cell_features,
        adjacency,
        frozen_access_embeddings,
        species_adjacency,
        torch.from_numpy(checklist_cell.astype(np.int64)),
        cell_species_support_features,
        test_checklists,
        species_indices,
        args.batch_size,
    )
    labels = load_positive_labels(graph_dir, test_checklists, species_indices)
    all_nodes = pd.read_parquet(
        graph_dir / "checklist_nodes.parquet",
        columns=["checklist_index", "x", "y"],
    )
    all_blocks = pd.Series(
        assign_spatial_blocks(all_nodes, blocks_per_dim),
        index=all_nodes["checklist_index"].to_numpy(),
    )
    nodes = all_nodes.set_index("checklist_index").loc[test_checklists].reset_index()
    blocks = all_blocks.loc[test_checklists].to_numpy(dtype=np.int64)

    rows = []
    for species_position, common_name in enumerate(species_names):
        for block_id in sorted(np.unique(blocks)):
            mask = blocks == block_id
            rows.append(
                summarize_group(
                    common_name,
                    int(block_id),
                    base_prob[mask, species_position],
                    full_prob[mask, species_position],
                    access_delta[mask, species_position]
                    if access_delta is not None
                    else None,
                    labels[mask, species_position],
                )
            )

    frame = pd.DataFrame(rows).sort_values(
        ["common_name", "full_minus_base_auprc"], ascending=[True, True]
    )
    output_csv = output_dir / "species_block_residuals.csv"
    frame.to_csv(output_csv, index=False)

    metadata_output = output_dir / "metadata.json"
    metadata_output.write_text(
        json.dumps(
            {
                "graph_dir": str(graph_dir),
                "run_name": args.run_name,
                "species": species_names,
                "blocks_per_dim": blocks_per_dim,
                "cell_metadata": cell_metadata,
                "species_graph_metadata": species_graph_metadata,
                "output": str(output_csv),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote species/block residual diagnostics to {output_dir}")
    print("\nLargest full-vs-base AUPRC losses:")
    print(
        frame.sort_values("full_minus_base_auprc")
        .head(15)
        .to_string(index=False, float_format="%.5f")
    )
    print("\nLargest positive suppression by block/species:")
    print(
        frame.sort_values("positive_mean_delta")
        .head(15)
        .to_string(index=False, float_format="%.5f")
    )


if __name__ == "__main__":
    main()
