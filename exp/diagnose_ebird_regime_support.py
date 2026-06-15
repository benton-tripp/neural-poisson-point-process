"""
Diagnose ecological/access regime support for held-out eBird spatial blocks.

This is a framework diagnostic. It asks whether held-out spatial blocks are
covered by analogous training cells in coastal/water/elevation/canopy/access
space. Repeated failures in a held-out block are less informative about model
architecture if that block is also ecologically or access-wise out of support.

Run from the project root:

    python exp/diagnose_ebird_regime_support.py --graph-dir data/ebird/graph_top100_spatial_10x10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_ebird_block_species import assign_spatial_blocks


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial_10x10"
ECOLOGY_COLUMNS = [
    "canopy_median",
    "nc_usgs30m_match_tcc",
    "distance_to_waterbody_m",
    "distance_to_coastline_m",
]
ACCESS_COLUMNS = [
    "duration_minutes",
    "effort_distance_km",
    "number_observers",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize held-out block support in ecology/access regime space."
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
            "graph-dir/spatial_gnn_baselines/diagnostics/regime_support."
        ),
    )
    parser.add_argument(
        "--spatial-grid-size-m",
        type=float,
        default=25_000.0,
        help="Grid cell size used for regime cells. Defaults to 25,000 m.",
    )
    parser.add_argument(
        "--min-test-cell-checklists",
        type=int,
        default=25,
        help="Minimum test checklists for cell-level support rows. Defaults to 25.",
    )
    parser.add_argument(
        "--coastal-distance-m",
        type=float,
        default=25_000.0,
        help="Cells/checklists within this coastline distance are coastal. Defaults to 25 km.",
    )
    parser.add_argument(
        "--near-water-distance-m",
        type=float,
        default=2_500.0,
        help="Cells/checklists within this waterbody distance are near water. Defaults to 2.5 km.",
    )
    parser.add_argument(
        "--nearest-train-cells",
        type=int,
        default=5,
        help="Number of nearest training cells used for support summaries. Defaults to 5.",
    )
    parser.add_argument(
        "--top-species",
        type=int,
        default=15,
        help="Top test-positive species rows per held-out block. Defaults to 15.",
    )
    return parser.parse_args()


def log1p_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame[columns].copy()
    for column in columns:
        if column.startswith("distance_to_") or column in {
            "duration_minutes",
            "effort_distance_km",
            "number_observers",
            "train_checklists",
            "test_checklists",
        }:
            out[column] = np.log1p(out[column].astype(float).clip(lower=0.0))
    return out.astype(float)


def standardized_distance(
    source: pd.DataFrame,
    target: pd.DataFrame,
    columns: list[str],
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if source.empty or target.empty:
        return (
            np.full(len(target), np.nan, dtype=np.float64),
            np.full(len(target), np.nan, dtype=np.float64),
        )
    source_values = log1p_columns(source, columns).to_numpy(dtype=np.float64)
    target_values = log1p_columns(target, columns).to_numpy(dtype=np.float64)
    mean = source_values.mean(axis=0, keepdims=True)
    std = source_values.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    source_z = (source_values - mean) / std
    target_z = (target_values - mean) / std
    distances = (
        np.sum(target_z**2, axis=1, keepdims=True)
        + np.sum(source_z**2, axis=1, keepdims=True).T
        - 2.0 * target_z @ source_z.T
    )
    distances = np.sqrt(np.maximum(distances, 0.0))
    nearest = np.partition(distances, kth=min(k, source_z.shape[0]) - 1, axis=1)[
        :, : min(k, source_z.shape[0])
    ]
    return nearest[:, 0], nearest.mean(axis=1)


def spatial_nearest_distance(source: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    if source.empty or target.empty:
        return np.full(len(target), np.nan, dtype=np.float64)
    source_xy = source[["x_mean", "y_mean"]].to_numpy(dtype=np.float64)
    target_xy = target[["x_mean", "y_mean"]].to_numpy(dtype=np.float64)
    distances = (
        np.sum(target_xy**2, axis=1, keepdims=True)
        + np.sum(source_xy**2, axis=1, keepdims=True).T
        - 2.0 * target_xy @ source_xy.T
    )
    return np.sqrt(np.maximum(distances, 0.0)).min(axis=1)


def assign_cells(nodes: pd.DataFrame, grid_size_m: float) -> np.ndarray:
    cell_x = np.floor(nodes["x"].to_numpy(dtype=np.float64) / grid_size_m).astype(np.int64)
    cell_y = np.floor(nodes["y"].to_numpy(dtype=np.float64) / grid_size_m).astype(np.int64)
    keys = pd.Series(cell_x.astype(str) + "_" + cell_y.astype(str))
    codes, _unique = pd.factorize(keys, sort=False)
    return codes.astype(np.int64)


def add_regime_columns(
    nodes: pd.DataFrame,
    cell_ids: np.ndarray,
    block_ids: np.ndarray,
    coastal_distance_m: float,
    near_water_distance_m: float,
) -> pd.DataFrame:
    work = nodes.copy()
    work["spatial_cell"] = cell_ids
    work["spatial_block"] = block_ids
    work["is_coastal"] = work["distance_to_coastline_m"] <= coastal_distance_m
    work["is_near_water"] = work["distance_to_waterbody_m"] <= near_water_distance_m
    work["is_stationary"] = work["protocol_name"] == "Stationary"
    work["is_traveling"] = work["protocol_name"] == "Traveling"
    return work


def summarize_work(
    work: pd.DataFrame,
    group_columns: list[str],
    train_count_column: str = "train_mask",
    test_count_column: str = "test_mask",
) -> pd.DataFrame:
    grouped = work.groupby(group_columns, sort=True)
    rows = grouped.agg(
        checklists=("checklist_index", "size"),
        train_checklists=(train_count_column, "sum"),
        test_checklists=(test_count_column, "sum"),
        x_mean=("x", "mean"),
        y_mean=("y", "mean"),
        coastal_rate=("is_coastal", "mean"),
        near_water_rate=("is_near_water", "mean"),
        stationary_rate=("is_stationary", "mean"),
        traveling_rate=("is_traveling", "mean"),
        unique_observers=("observer_id", "nunique"),
        unique_localities=("locality_id", "nunique"),
        canopy_median=("canopy_median", "mean"),
        nc_usgs30m_match_tcc=("nc_usgs30m_match_tcc", "mean"),
        distance_to_waterbody_m=("distance_to_waterbody_m", "mean"),
        distance_to_coastline_m=("distance_to_coastline_m", "mean"),
        duration_minutes=("duration_minutes", "mean"),
        effort_distance_km=("effort_distance_km", "mean"),
        number_observers=("number_observers", "mean"),
    ).reset_index()
    if "spatial_block" not in rows.columns:
        block_mode = (
            grouped["spatial_block"]
            .agg(lambda x: int(pd.Series.mode(x).iloc[0]))
            .rename("spatial_block")
            .reset_index()
        )
        rows = rows.merge(block_mode, on=group_columns, how="left")
    rows["observer_per_checklist"] = rows["unique_observers"] / rows["checklists"].clip(lower=1)
    rows["locality_per_checklist"] = rows["unique_localities"] / rows["checklists"].clip(lower=1)
    return rows


def summarize_cells(
    nodes: pd.DataFrame,
    cell_ids: np.ndarray,
    block_ids: np.ndarray,
    coastal_distance_m: float,
    near_water_distance_m: float,
) -> pd.DataFrame:
    work = add_regime_columns(
        nodes,
        cell_ids,
        block_ids,
        coastal_distance_m,
        near_water_distance_m,
    )
    return summarize_work(work, ["spatial_cell"])


def summarize_train_test_cells(
    nodes: pd.DataFrame,
    cell_ids: np.ndarray,
    block_ids: np.ndarray,
    coastal_distance_m: float,
    near_water_distance_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = add_regime_columns(
        nodes,
        cell_ids,
        block_ids,
        coastal_distance_m,
        near_water_distance_m,
    )

    train_work = work.loc[work["train_mask"]].copy()
    train_work["train_row"] = True
    train_work["test_row"] = False
    train_cells = summarize_work(
        train_work,
        ["spatial_cell"],
        train_count_column="train_row",
        test_count_column="test_row",
    )

    test_work = work.loc[work["test_mask"]].copy()
    test_work["train_row"] = False
    test_work["test_row"] = True
    # Group test rows by both cell and validation block. A 25 km regime cell can
    # straddle a validation block boundary; for held-out diagnostics the block
    # label must come from the held-out checklists, not the all-row modal block.
    test_cells = summarize_work(
        test_work,
        ["spatial_block", "spatial_cell"],
        train_count_column="train_row",
        test_count_column="test_row",
    )
    return train_cells, test_cells


def add_species_support(
    graph_dir: Path,
    nodes: pd.DataFrame,
    block_ids: np.ndarray,
    output_dir: Path,
    top_species: int,
) -> pd.DataFrame:
    species = pd.read_csv(graph_dir / "species_nodes.csv")
    positive_edges = pd.read_parquet(graph_dir / "positive_edges.parquet")
    block_lookup = pd.DataFrame(
        {
            "checklist_index": nodes["checklist_index"].to_numpy(dtype=np.int64),
            "spatial_block": block_ids.astype(np.int64),
            "train_mask": nodes["train_mask"].to_numpy(dtype=bool),
            "test_mask": nodes["test_mask"].to_numpy(dtype=bool),
        }
    )
    edges = positive_edges.merge(block_lookup, on="checklist_index", how="left")
    train_counts = (
        edges.loc[edges["train_mask"]]
        .groupby("species_index")
        .size()
        .rename("train_positive_checklists")
    )
    test_counts = (
        edges.loc[edges["test_mask"]]
        .groupby(["spatial_block", "species_index"])
        .size()
        .rename("test_positive_checklists")
        .reset_index()
    )
    block_sizes = (
        block_lookup.loc[block_lookup["test_mask"]]
        .groupby("spatial_block")
        .size()
        .rename("test_checklists")
    )
    rows = test_counts.merge(block_sizes, on="spatial_block", how="left")
    rows = rows.merge(train_counts, on="species_index", how="left")
    rows["train_positive_checklists"] = rows["train_positive_checklists"].fillna(0).astype(int)
    rows["test_prevalence"] = rows["test_positive_checklists"] / rows["test_checklists"]
    rows = rows.merge(
        species[["species_index", "species_key", "common_name", "scientific_name"]],
        on="species_index",
        how="left",
    )
    rows = rows.sort_values(
        ["spatial_block", "test_positive_checklists"],
        ascending=[True, False],
    )
    top_rows = rows.groupby("spatial_block", group_keys=False).head(top_species)
    top_rows.to_csv(output_dir / "regime_support_top_species_by_test_block.csv", index=False)
    return rows


def block_summary(test_cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for block, group in test_cells.groupby("spatial_block", sort=True):
        weights = group["test_checklists"].to_numpy(dtype=np.float64)
        weights = weights / weights.sum()
        row = {
            "spatial_block": int(block),
            "test_cells": int(len(group)),
            "test_checklists": int(group["test_checklists"].sum()),
        }
        for column in [
            "coastal_rate",
            "near_water_rate",
            "stationary_rate",
            "traveling_rate",
            "canopy_median",
            "nc_usgs30m_match_tcc",
            "distance_to_waterbody_m",
            "distance_to_coastline_m",
            "duration_minutes",
            "effort_distance_km",
            "number_observers",
            "nearest_train_ecology_distance",
            "nearest5_train_ecology_distance",
            "nearest_train_access_distance",
            "nearest5_train_access_distance",
            "nearest_train_spatial_distance_m",
        ]:
            if column in group.columns:
                row[column] = float(np.sum(group[column].to_numpy(dtype=np.float64) * weights))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["nearest_train_ecology_distance", "test_checklists"],
        ascending=[False, False],
    )


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else graph_dir / "spatial_gnn_baselines" / "diagnostics" / "regime_support"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    nodes = pd.read_parquet(graph_dir / "checklist_nodes.parquet").sort_values(
        "checklist_index"
    )
    missing = set(ECOLOGY_COLUMNS + ACCESS_COLUMNS).difference(nodes.columns)
    if missing:
        raise ValueError(f"checklist_nodes.parquet is missing required columns: {sorted(missing)}")

    blocks_per_dim = int(metadata.get("split", {}).get("spatial_blocks_per_dim", 10))
    block_ids = assign_spatial_blocks(nodes, blocks_per_dim)
    cell_ids = assign_cells(nodes, args.spatial_grid_size_m)
    cells = summarize_cells(
        nodes,
        cell_ids,
        block_ids,
        args.coastal_distance_m,
        args.near_water_distance_m,
    )
    train_cells, test_cells = summarize_train_test_cells(
        nodes,
        cell_ids,
        block_ids,
        args.coastal_distance_m,
        args.near_water_distance_m,
    )
    test_cells = test_cells[
        test_cells["test_checklists"] >= args.min_test_cell_checklists
    ].copy()

    ecology_min, ecology_k = standardized_distance(
        train_cells,
        test_cells,
        ECOLOGY_COLUMNS,
        args.nearest_train_cells,
    )
    access_min, access_k = standardized_distance(
        train_cells,
        test_cells,
        ACCESS_COLUMNS + ["stationary_rate", "traveling_rate"],
        args.nearest_train_cells,
    )
    test_cells["nearest_train_ecology_distance"] = ecology_min
    test_cells["nearest5_train_ecology_distance"] = ecology_k
    test_cells["nearest_train_access_distance"] = access_min
    test_cells["nearest5_train_access_distance"] = access_k
    test_cells["nearest_train_spatial_distance_m"] = spatial_nearest_distance(
        train_cells, test_cells
    )

    summary = block_summary(test_cells)
    cells.to_csv(output_dir / "regime_support_all_cells.csv", index=False)
    test_cells.to_csv(output_dir / "regime_support_test_cells.csv", index=False)
    summary.to_csv(output_dir / "regime_support_test_block_summary.csv", index=False)
    species_rows = add_species_support(
        graph_dir,
        nodes,
        block_ids,
        output_dir,
        args.top_species,
    )

    run_metadata = {
        "graph_dir": str(graph_dir),
        "spatial_grid_size_m": args.spatial_grid_size_m,
        "coastal_distance_m": args.coastal_distance_m,
        "near_water_distance_m": args.near_water_distance_m,
        "nearest_train_cells": args.nearest_train_cells,
        "min_test_cell_checklists": args.min_test_cell_checklists,
        "spatial_blocks_per_dim": blocks_per_dim,
        "test_blocks": metadata.get("split", {}).get("test_blocks_ids", []),
        "outputs": {
            "all_cells": "regime_support_all_cells.csv",
            "test_cells": "regime_support_test_cells.csv",
            "test_block_summary": "regime_support_test_block_summary.csv",
            "top_species_by_test_block": "regime_support_top_species_by_test_block.csv",
        },
    }
    (output_dir / "regime_support_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )

    display_columns = [
        "spatial_block",
        "test_cells",
        "test_checklists",
        "coastal_rate",
        "near_water_rate",
        "distance_to_coastline_m",
        "distance_to_waterbody_m",
        "nearest_train_ecology_distance",
        "nearest_train_access_distance",
        "nearest_train_spatial_distance_m",
    ]
    print(f"Wrote regime-support diagnostics to {output_dir}")
    print("\nHeld-out block regime/support summary:")
    print(summary[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    top_species_rows = species_rows.groupby("spatial_block", group_keys=False).head(5)
    if not top_species_rows.empty:
        print("\nTop held-out positive species per test block:")
        print(
            top_species_rows[
                [
                    "spatial_block",
                    "common_name",
                    "test_positive_checklists",
                    "test_prevalence",
                    "train_positive_checklists",
                ]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )


if __name__ == "__main__":
    main()
