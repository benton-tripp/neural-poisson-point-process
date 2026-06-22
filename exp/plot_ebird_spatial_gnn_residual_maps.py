"""
Plot spatial GNN residual/probability-difference maps for selected species.

The script compares each saved spatial GNN's full prediction with its own base
checklist/species path before the spatial residual is added. This isolates the
direction and magnitude of the message-passing spatial correction.

Run from the project root:

    python exp/plot_ebird_spatial_gnn_residual_maps.py --graph-dir data/ebird/graph_top100_spatial --run-name spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pandas as pd
import torch

from ebird_graph_all_species_baseline import build_label_matrix, load_split_checklists
from ebird_spatial_gnn_baseline import (
    SpatialGCNHybrid,
    build_cell_species_support_features,
    build_species_adjacency_for_run,
    build_spatial_cell_graph_for_run,
    inject_frozen_access_embeddings,
    load_frozen_access_embeddings,
)
from ebird_joint_tabular_baseline import SEED


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial"
DEFAULT_RUN_NAME = "spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001"
DEFAULT_BOUNDARY = "data/boundaries/nc_state_boundary.gpkg"
DEFAULT_SPECIES = [
    "Black-and-white Warbler",
    "Eastern Meadowlark",
    "Red-headed Woodpecker",
    "Swamp Sparrow",
    "Wood Duck",
    "Green Heron",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot spatial GNN residual probability maps for selected species."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help=f"Spatial GNN run name without the spatial_gnn_ prefix. Defaults to {DEFAULT_RUN_NAME}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to graph-dir/spatial_gnn_baselines/diagnostics/residual_maps.",
    )
    parser.add_argument(
        "--species",
        nargs="*",
        default=DEFAULT_SPECIES,
        help="Species common names to plot.",
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
    parser.add_argument(
        "--boundary",
        default=DEFAULT_BOUNDARY,
        help=f"Optional boundary layer to draw on maps. Defaults to {DEFAULT_BOUNDARY}.",
    )
    parser.add_argument(
        "--map-crs",
        default="EPSG:5070",
        help="CRS for checklist coordinates and map plotting. Defaults to EPSG:5070.",
    )
    return parser.parse_args()


def load_boundary(path: str | None, map_crs: str) -> gpd.GeoDataFrame | None:
    if not path:
        return None
    boundary_path = Path(path)
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file does not exist: {boundary_path}")
    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        boundary = boundary.set_crs(map_crs)
    elif str(boundary.crs) != map_crs:
        boundary = boundary.to_crs(map_crs)
    return boundary


def draw_boundary(ax: plt.Axes, boundary: gpd.GeoDataFrame | None) -> None:
    if boundary is not None and not boundary.empty:
        boundary.boundary.plot(ax=ax, color="#1F1F1F", linewidth=0.8, zorder=5)


def load_run(graph_dir: Path, run_name: str) -> tuple[dict, Path]:
    output_dir = graph_dir / "spatial_gnn_baselines"
    prefix = f"spatial_gnn_{run_name}"
    summary_path = output_dir / f"{prefix}_summary.json"
    model_path = output_dir / f"{prefix}_model.pt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary JSON: {summary_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model state: {model_path}")
    return json.loads(summary_path.read_text(encoding="utf-8")), model_path


def base_and_full_logits(
    model: SpatialGCNHybrid,
    checklist_features: torch.Tensor,
    cell_embeddings: torch.Tensor,
    checklist_cells: torch.Tensor,
    cell_species_support_features: torch.Tensor | None,
    species_adjacency: torch.Tensor | None = None,
    cell_features: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model.gnn_mode == "concat":
        raise ValueError("Residual maps require a residual or gated spatial GNN.")
    return model.base_and_full_logits(
        checklist_features,
        cell_embeddings,
        checklist_cells,
        species_adjacency,
        cell_features=cell_features,
        cell_species_support_features=cell_species_support_features,
    )


def predict_selected_species(
    model: SpatialGCNHybrid,
    features: torch.Tensor,
    cell_features: torch.Tensor,
    adjacency: torch.Tensor,
    frozen_access_embeddings: torch.Tensor | None,
    species_adjacency: torch.Tensor | None,
    checklist_cells: torch.Tensor,
    cell_species_support_features: torch.Tensor | None,
    checklist_indices: np.ndarray,
    species_indices: list[int],
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    model.eval()
    base_parts = []
    full_parts = []
    access_delta_parts = []
    species_tensor = torch.tensor(species_indices, dtype=torch.int64)
    with torch.no_grad():
        cell_embeddings = model.encode_cells(cell_features, adjacency)
        cell_embeddings = inject_frozen_access_embeddings(
            model, cell_embeddings, frozen_access_embeddings
        )
        for start in range(0, len(checklist_indices), batch_size):
            batch = checklist_indices[start : start + batch_size]
            batch_tensor = torch.from_numpy(batch.astype(np.int64))
            base_logits, full_logits = base_and_full_logits(
                model,
                features[batch_tensor],
                cell_embeddings,
                checklist_cells[batch_tensor],
                species_adjacency=species_adjacency,
                cell_features=cell_features,
                cell_species_support_features=cell_species_support_features,
            )
            base_scores = torch.sigmoid(base_logits[:, species_tensor])
            base_parts.append(base_scores.cpu().numpy())
            full_parts.append(torch.sigmoid(full_logits[:, species_tensor]).cpu().numpy())
            if model.last_spatial_access_bias_logits is not None:
                access_logits = model.last_spatial_access_bias_logits
                no_access_scores = torch.sigmoid(
                    (base_logits - access_logits)[:, species_tensor]
                )
                access_delta_parts.append((base_scores - no_access_scores).cpu().numpy())
    access_delta = (
        np.vstack(access_delta_parts).astype(np.float32)
        if access_delta_parts
        else None
    )
    return (
        np.vstack(base_parts).astype(np.float32),
        np.vstack(full_parts).astype(np.float32),
        access_delta,
    )


def load_positive_labels(
    graph_dir: Path,
    checklist_indices: np.ndarray,
    species_indices: list[int],
) -> np.ndarray:
    edges = pd.read_parquet(
        graph_dir / "positive_edges.parquet",
        columns=["checklist_index", "species_index", "split"],
    )
    edges = edges.loc[
        (edges["split"] == "test") & (edges["species_index"].isin(species_indices))
    ]
    checklist_lookup = pd.Series(
        np.arange(len(checklist_indices), dtype=np.int64),
        index=checklist_indices,
    )
    species_lookup = {species_index: i for i, species_index in enumerate(species_indices)}
    labels = np.zeros((len(checklist_indices), len(species_indices)), dtype=bool)
    rows = edges["checklist_index"].map(checklist_lookup).to_numpy()
    cols = edges["species_index"].map(species_lookup).to_numpy()
    valid = pd.notna(rows) & pd.notna(cols)
    labels[rows[valid].astype(np.int64), cols[valid].astype(np.int64)] = True
    return labels


def safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def plot_species_delta(
    nodes: pd.DataFrame,
    common_name: str,
    base_prob: np.ndarray,
    full_prob: np.ndarray,
    access_delta: np.ndarray | None,
    labels: np.ndarray,
    output: Path,
    boundary: gpd.GeoDataFrame | None,
) -> dict:
    delta = full_prob - base_prob
    limit = float(np.nanpercentile(np.abs(delta), 98))
    if not np.isfinite(limit) or limit <= 0:
        limit = float(np.max(np.abs(delta))) if len(delta) else 0.01
    limit = max(limit, 0.01)

    panel_count = 3 if access_delta is not None else 2
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(5.8 * panel_count, 5.5),
        sharex=True,
        sharey=True,
    )
    scatter = axes[0].scatter(
        nodes["x"],
        nodes["y"],
        c=delta,
        s=3,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        linewidths=0,
    )
    axes[0].set_title("Spatial residual probability delta")
    if access_delta is not None:
        access_limit = float(np.nanpercentile(np.abs(access_delta), 98))
        if not np.isfinite(access_limit) or access_limit <= 0:
            access_limit = float(np.max(np.abs(access_delta))) if len(access_delta) else 0.01
        access_limit = max(access_limit, 0.01)
        access_scatter = axes[1].scatter(
            nodes["x"],
            nodes["y"],
            c=access_delta,
            s=3,
            cmap="RdBu_r",
            vmin=-access_limit,
            vmax=access_limit,
            linewidths=0,
        )
        axes[1].set_title("Shared access probability delta")
        positive_axis = axes[2]
    else:
        access_scatter = None
        positive_axis = axes[1]
    positives = nodes.loc[labels]
    positive_axis.scatter(nodes["x"], nodes["y"], s=1, c="#D0D0D0", alpha=0.25, linewidths=0)
    positive_axis.scatter(positives["x"], positives["y"], s=4, c="#D62728", alpha=0.7, linewidths=0)
    positive_axis.set_title("Held-out positives")
    for ax in axes:
        draw_boundary(ax, boundary)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(common_name)
    fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04, label="full - base probability")
    if access_scatter is not None:
        fig.colorbar(
            access_scatter,
            ax=axes[1],
            fraction=0.046,
            pad=0.04,
            label="base - no-access probability",
        )
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close(fig)

    row = {
        "common_name": common_name,
        "test_checklists": int(len(nodes)),
        "positive_checklists": int(labels.sum()),
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
    return row


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else graph_dir / "spatial_gnn_baselines" / "diagnostics" / "residual_maps"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary = load_boundary(args.boundary, args.map_crs)

    summary, model_path = load_run(graph_dir, args.run_name)
    model_info = summary["model"]
    if model_info["gnn_mode"] not in {"residual", "gated"}:
        raise ValueError("Residual maps require --gnn-mode residual or gated.")

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
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)

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
    nodes = pd.read_parquet(
        graph_dir / "checklist_nodes.parquet",
        columns=["checklist_index", "x", "y", "protocol_name", "duration_minutes", "effort_distance_km"],
    )
    nodes = nodes.set_index("checklist_index").loc[test_checklists].reset_index()

    summary_rows = []
    for position, common_name in enumerate(species_names):
        summary_rows.append(
            plot_species_delta(
                nodes,
                common_name,
                base_prob[:, position],
                full_prob[:, position],
                access_delta[:, position] if access_delta is not None else None,
                labels[:, position],
                output_dir / f"{safe_name(common_name)}_residual_probability_delta.png",
                boundary,
            )
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output_dir / f"{args.run_name}_residual_probability_summary.csv", index=False)
    (output_dir / f"{args.run_name}_metadata.json").write_text(
        json.dumps(
            {
                "graph_dir": str(graph_dir),
                "run_name": args.run_name,
                "model_path": str(model_path),
                "species": species_names,
                "cell_metadata": cell_metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote residual maps to {output_dir}")
    print(summary_frame.to_string(index=False, float_format="%.5f"))


if __name__ == "__main__":
    main()
