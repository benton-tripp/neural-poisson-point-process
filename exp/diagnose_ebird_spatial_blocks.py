"""
Profile held-out spatial blocks for eBird graph validation.

This diagnostic explains whether a difficult held-out block is unusual in
effort/access, ecology, species composition, or access-encoder target space.

Run from the project root:

    python exp/diagnose_ebird_spatial_blocks.py --graph-dir data/ebird/graph_top100_spatial_10x10 --access-run-name access_gcn_h64_l2_z64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_ebird_block_species import assign_spatial_blocks
from ebird_graph_all_species_baseline import build_label_matrix, load_split_checklists
from ebird_joint_tabular_baseline import SEED
from ebird_spatial_gnn_baseline import build_spatial_cell_graph


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial_10x10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile spatial validation blocks by effort, ecology, and species."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            "graph-dir/spatial_gnn_baselines/diagnostics/block_profile."
        ),
    )
    parser.add_argument(
        "--access-run-name",
        default=None,
        help=(
            "Optional access-encoder run name under graph-dir/access_encoder. "
            "When provided, access target/prediction summaries are added."
        ),
    )
    parser.add_argument(
        "--spatial-grid-size-m",
        type=float,
        default=25_000.0,
        help="Spatial-cell size used for access encoder outputs. Defaults to 25,000 m.",
    )
    parser.add_argument(
        "--top-species",
        type=int,
        default=15,
        help="Species rows to print per held-out block. Defaults to 15.",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed. Defaults to 19.")
    return parser.parse_args()


def safe_mean(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean())


def summarize_block(group: pd.DataFrame, split: str, block_id: int) -> dict:
    protocol = group["protocol_name"].astype("string")
    species_count = group["species_count"].to_numpy(dtype=np.float64)
    return {
        "spatial_block": int(block_id),
        "split": split,
        "checklists": int(len(group)),
        "x_mean": safe_mean(group["x"]),
        "y_mean": safe_mean(group["y"]),
        "species_per_checklist_mean": float(np.mean(species_count)),
        "species_per_checklist_p90": float(np.quantile(species_count, 0.90)),
        "traveling_rate": float((protocol == "Traveling").mean()),
        "stationary_rate": float((protocol == "Stationary").mean()),
        "duration_minutes_mean": safe_mean(group["duration_minutes"]),
        "duration_minutes_p90": float(
            pd.to_numeric(group["duration_minutes"], errors="coerce").quantile(0.90)
        ),
        "effort_distance_km_mean": safe_mean(group["effort_distance_km"]),
        "effort_distance_km_p90": float(
            pd.to_numeric(group["effort_distance_km"], errors="coerce").quantile(0.90)
        ),
        "number_observers_mean": safe_mean(group["number_observers"]),
        "unique_observers": int(group["observer_id"].nunique()),
        "unique_localities": int(group["locality_id"].nunique()),
        "unique_counties": int(group["county"].nunique()),
        "observer_per_checklist": float(group["observer_id"].nunique() / max(len(group), 1)),
        "locality_per_checklist": float(group["locality_id"].nunique() / max(len(group), 1)),
        "canopy_median_mean": safe_mean(group["canopy_median"]),
        "elevation_mean": safe_mean(group["nc_usgs30m_match_tcc"]),
        "distance_to_waterbody_m_mean": safe_mean(group["distance_to_waterbody_m"]),
        "distance_to_coastline_m_mean": safe_mean(group["distance_to_coastline_m"]),
    }


def add_standardized_differences(summary: pd.DataFrame) -> pd.DataFrame:
    numeric = summary.select_dtypes(include=[np.number]).columns.tolist()
    numeric = [col for col in numeric if col != "spatial_block"]
    train = summary.loc[summary["split"] == "train"]
    out = summary.copy()
    for col in numeric:
        mean = train[col].mean()
        std = train[col].std()
        if not np.isfinite(std) or std == 0:
            out[f"{col}_z_vs_train_blocks"] = np.nan
        else:
            out[f"{col}_z_vs_train_blocks"] = (out[col] - mean) / std
    return out


def species_prevalence_rows(
    graph_dir: Path,
    block_nodes: pd.DataFrame,
    split_indices: np.ndarray,
    split: str,
    species_count: int,
    species: pd.DataFrame,
) -> pd.DataFrame:
    labels = build_label_matrix(graph_dir, split_indices, species_count, split)
    rows = []
    for block_id in sorted(block_nodes["spatial_block"].unique()):
        mask = block_nodes["spatial_block"].to_numpy() == block_id
        if not mask.any():
            continue
        block_labels = labels[mask]
        positives = block_labels.sum(axis=0)
        prevalence = positives / max(block_labels.shape[0], 1)
        for species_row in species.itertuples(index=False):
            species_index = int(species_row.species_index)
            rows.append(
                {
                    "spatial_block": int(block_id),
                    "split": split,
                    "common_name": species_row.common_name,
                    "scientific_name": species_row.scientific_name,
                    "checklists": int(block_labels.shape[0]),
                    "positive_checklists": int(positives[species_index]),
                    "prevalence": float(prevalence[species_index]),
                }
            )
    return pd.DataFrame(rows)


def block_species_distance(species_prev: pd.DataFrame) -> pd.DataFrame:
    pivot = species_prev.pivot_table(
        index=["split", "spatial_block"],
        columns="common_name",
        values="prevalence",
        fill_value=0.0,
    )
    train = pivot.loc["train"] if "train" in pivot.index.get_level_values(0) else pd.DataFrame()
    test = pivot.loc["test"] if "test" in pivot.index.get_level_values(0) else pd.DataFrame()
    rows = []
    if train.empty or test.empty:
        return pd.DataFrame(rows)
    train_matrix = train.to_numpy(dtype=np.float64)
    for test_block, test_values in test.iterrows():
        diff = train_matrix - test_values.to_numpy(dtype=np.float64)[None, :]
        distances = np.sqrt((diff**2).sum(axis=1))
        nearest_pos = int(np.argmin(distances))
        rows.append(
            {
                "test_spatial_block": int(test_block),
                "nearest_train_spatial_block": int(train.index[nearest_pos]),
                "species_prevalence_l2_to_nearest_train": float(distances[nearest_pos]),
                "species_prevalence_l2_to_train_mean": float(
                    np.sqrt(
                        (
                            (test_values.to_numpy(dtype=np.float64) - train_matrix.mean(axis=0))
                            ** 2
                        ).sum()
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def load_access_encoder_summary(
    graph_dir: Path,
    access_run_name: str | None,
    nodes: pd.DataFrame,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if not access_run_name:
        return None, None
    access_dir = graph_dir / "access_encoder"
    target_path = access_dir / f"{access_run_name}_cell_access_targets.csv"
    prediction_path = access_dir / f"{access_run_name}_cell_predictions_z.npy"
    summary_path = access_dir / f"{access_run_name}_summary.json"
    if not target_path.exists() or not prediction_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "Missing access encoder outputs. Expected target CSV, prediction NPY, "
            f"and summary JSON for run {access_run_name} in {access_dir}."
        )
    access_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    target_names = access_summary["model"]["target_names"]
    targets = pd.read_csv(target_path)
    predictions = np.load(prediction_path).astype(np.float32)
    prediction_columns = [f"{name}_pred_z" for name in target_names]
    pred_frame = pd.DataFrame(predictions, columns=prediction_columns)
    pred_frame["spatial_cell"] = np.arange(len(pred_frame), dtype=np.int64)
    access = targets.merge(pred_frame, on="spatial_cell", how="left")
    cell_blocks = (
        nodes[["spatial_cell", "spatial_block"]]
        .drop_duplicates()
        .groupby("spatial_cell")["spatial_block"]
        .agg(lambda values: int(pd.Series(values).mode().iloc[0]))
        .reset_index()
    )
    access = access.merge(cell_blocks, on="spatial_cell", how="left")
    rows = []
    for block_id, group in access.groupby("spatial_block", dropna=True):
        row = {"spatial_block": int(block_id)}
        for column in access.columns:
            if column in {"spatial_cell", "spatial_block"}:
                continue
            if pd.api.types.is_numeric_dtype(access[column]):
                row[f"access_{column}_mean"] = float(group[column].mean())
        rows.append(row)
    block_access = pd.DataFrame(rows)
    metrics_path = access_dir / f"{access_run_name}_target_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.exists() else None
    return block_access, metrics


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else graph_dir / "spatial_gnn_baselines" / "diagnostics" / "block_profile"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    features_np = np.load(graph_dir / "checklist_features.npy").astype(np.float32)
    species_count = int(metadata["counts"]["species"])
    blocks_per_dim = int(metadata.get("split", {}).get("spatial_blocks_per_dim", 10))
    species = pd.read_csv(graph_dir / "species_nodes.csv")
    train_checklists = load_split_checklists(graph_dir, "train", None, args.seed)
    test_checklists = load_split_checklists(graph_dir, "test", None, args.seed)
    train_set = set(train_checklists.tolist())
    test_set = set(test_checklists.tolist())

    nodes = pd.read_parquet(graph_dir / "checklist_nodes.parquet").sort_values(
        "checklist_index"
    )
    nodes["spatial_block"] = assign_spatial_blocks(nodes, blocks_per_dim)
    nodes["split"] = np.where(
        nodes["checklist_index"].isin(test_set),
        "test",
        np.where(nodes["checklist_index"].isin(train_set), "train", "unused"),
    )
    train_labels = build_label_matrix(graph_dir, train_checklists, species_count, "train")
    test_labels = build_label_matrix(graph_dir, test_checklists, species_count, "test")
    _cell_features, _adjacency, checklist_cell, _cell_metadata = build_spatial_cell_graph(
        graph_dir,
        features_np,
        train_checklists,
        args.spatial_grid_size_m,
    )
    species_count_by_check = np.zeros(len(nodes), dtype=np.float32)
    species_count_by_check[train_checklists] = train_labels.sum(axis=1)
    species_count_by_check[test_checklists] = test_labels.sum(axis=1)
    nodes["species_count"] = species_count_by_check
    nodes["spatial_cell"] = checklist_cell

    block_rows = []
    for (split, block_id), group in nodes.loc[nodes["split"].isin(["train", "test"])].groupby(
        ["split", "spatial_block"]
    ):
        block_rows.append(summarize_block(group, split, int(block_id)))
    block_summary = add_standardized_differences(pd.DataFrame(block_rows))

    train_nodes = nodes.loc[nodes["checklist_index"].isin(train_set)].copy()
    test_nodes = nodes.loc[nodes["checklist_index"].isin(test_set)].copy()
    species_prev = pd.concat(
        [
            species_prevalence_rows(
                graph_dir, train_nodes, train_checklists, "train", species_count, species
            ),
            species_prevalence_rows(
                graph_dir, test_nodes, test_checklists, "test", species_count, species
            ),
        ],
        ignore_index=True,
    )
    species_distance = block_species_distance(species_prev)
    block_access, access_metrics = load_access_encoder_summary(
        graph_dir, args.access_run_name, nodes
    )
    if block_access is not None:
        block_summary = block_summary.merge(block_access, on="spatial_block", how="left")

    block_summary.to_csv(output_dir / "block_profile_summary.csv", index=False)
    species_prev.to_csv(output_dir / "block_species_prevalence.csv", index=False)
    species_distance.to_csv(output_dir / "test_block_species_distance.csv", index=False)
    if access_metrics is not None:
        access_metrics.to_csv(output_dir / "access_encoder_target_metrics.csv", index=False)
    (output_dir / "block_profile_metadata.json").write_text(
        json.dumps(
            {
                "graph_dir": str(graph_dir),
                "access_run_name": args.access_run_name,
                "blocks_per_dim": blocks_per_dim,
                "train_checklists": int(len(train_checklists)),
                "test_checklists": int(len(test_checklists)),
                "outputs": {
                    "block_profile_summary": "block_profile_summary.csv",
                    "block_species_prevalence": "block_species_prevalence.csv",
                    "test_block_species_distance": "test_block_species_distance.csv",
                    "access_encoder_target_metrics": "access_encoder_target_metrics.csv"
                    if access_metrics is not None
                    else None,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    test_blocks = block_summary.loc[block_summary["split"] == "test"].copy()
    key_cols = [
        "spatial_block",
        "checklists",
        "species_per_checklist_mean",
        "traveling_rate",
        "duration_minutes_mean",
        "effort_distance_km_mean",
        "number_observers_mean",
        "observer_per_checklist",
        "locality_per_checklist",
        "canopy_median_mean",
        "elevation_mean",
        "distance_to_waterbody_m_mean",
        "distance_to_coastline_m_mean",
    ]
    print(f"Wrote block profile diagnostics to {output_dir}")
    print("\nHeld-out block profile:")
    print(test_blocks[key_cols].to_string(index=False, float_format="%.4f"))
    z_cols = [col for col in test_blocks.columns if col.endswith("_z_vs_train_blocks")]
    z_display = ["spatial_block"] + sorted(
        z_cols,
        key=lambda col: float(test_blocks[col].abs().max(skipna=True)),
        reverse=True,
    )[:12]
    print("\nLargest held-out z-scores versus train blocks:")
    print(test_blocks[z_display].to_string(index=False, float_format="%.4f"))
    if not species_distance.empty:
        print("\nSpecies-composition distance to nearest train block:")
        print(species_distance.to_string(index=False, float_format="%.4f"))
    print("\nTop species by held-out block prevalence:")
    top_prev = (
        species_prev.loc[species_prev["split"] == "test"]
        .sort_values(["spatial_block", "prevalence"], ascending=[True, False])
        .groupby("spatial_block")
        .head(args.top_species)
    )
    print(
        top_prev[
            [
                "spatial_block",
                "common_name",
                "positive_checklists",
                "prevalence",
            ]
        ].to_string(index=False, float_format="%.4f")
    )


if __name__ == "__main__":
    main()
