"""
Build species-level diagnostics for spatial GNN gain/loss species.

Run from the project root after graph-vs-tabular comparison CSVs are available:

    python exp/diagnose_ebird_spatial_gnn_species.py --graph-dir data/ebird/graph_top100_spatial --comparison-csv data/ebird/graph_top100_spatial/spatial_gnn_baselines/top100_mlp_both_spatial-stratified_spatial_gnn_spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001_test_species_metrics_graph_vs_tabular_species.csv
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


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial"
DEFAULT_BOUNDARY = "data/boundaries/nc_state_boundary.gpkg"
DEFAULT_FAILURE_SPECIES = [
    "Red-headed Woodpecker",
    "Swamp Sparrow",
    "Wood Duck",
    "Green Heron",
    "Ovenbird",
]
DEFAULT_GAIN_SPECIES = [
    "Black-and-white Warbler",
    "Double-crested Cormorant",
    "Mallard",
    "Yellow-billed Cuckoo",
    "Eastern Meadowlark",
]
ECOLOGY_COLUMNS = [
    "canopy_median",
    "nc_usgs30m_match_tcc",
    "distance_to_waterbody_m",
    "distance_to_coastline_m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose species-level spatial GNN gains/losses."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--comparison-csv",
        action="append",
        default=[],
        help=(
            "Graph-vs-tabular species comparison CSV. Repeat for multiple models. "
            "If omitted, recent comparison CSVs in graph-dir/spatial_gnn_baselines are used."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to graph-dir/spatial_gnn_baselines/diagnostics/species_diagnostics.",
    )
    parser.add_argument(
        "--focus-species",
        nargs="*",
        default=None,
        help="Optional common names to diagnose. Defaults to major gain/loss species.",
    )
    parser.add_argument(
        "--top-delta-species",
        type=int,
        default=8,
        help="Top positive and negative AUPRC-delta species to add from each comparison. Defaults to 8.",
    )
    parser.add_argument(
        "--spatial-grid-size-m",
        type=float,
        default=25_000.0,
        help="Spatial cell size for coverage diagnostics. Defaults to 25,000 m.",
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


def short_model_label(path: Path) -> str:
    stem = path.stem
    if "gated" in stem:
        if "gbm2" in stem:
            return "gated_gbm2"
        if "gbm3" in stem:
            return "gated_gbm3"
        return "gated"
    if "residual" in stem:
        if "cl1_wd0p0001" in stem:
            return "residual_primary"
        if "cl1_wd0p001" in stem:
            return "residual_wd0p001"
        return "residual"
    if "spatial_gcn" in stem:
        return "concat_gcn"
    return stem[:48]


def discover_comparison_csvs(graph_dir: Path) -> list[Path]:
    output_dir = graph_dir / "spatial_gnn_baselines"
    paths = sorted(output_dir.glob("top*_mlp_*spatial_gnn*.csv"))
    return [
        path
        for path in paths
        if "test_species_metrics" in path.name or "graph_vs" in path.name
    ]


def load_comparisons(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {
            "graph_common_name",
            "graph_minus_tabular_auprc",
            "graph_minus_tabular_auroc",
            "graph_calibration_error",
            "tabular_calibration_error",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        frame["model_label"] = short_model_label(path)
        frame["comparison_csv"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No graph-vs-tabular comparison CSVs found.")
    return pd.concat(frames, ignore_index=True)


def choose_focus_species(comparison: pd.DataFrame, explicit: list[str] | None, top_n: int) -> list[str]:
    if explicit:
        return list(dict.fromkeys(explicit))
    names = DEFAULT_FAILURE_SPECIES + DEFAULT_GAIN_SPECIES
    for _, frame in comparison.groupby("model_label", sort=False):
        names.extend(
            frame.sort_values("graph_minus_tabular_auprc", ascending=False)
            .head(top_n)["graph_common_name"]
            .tolist()
        )
        names.extend(
            frame.sort_values("graph_minus_tabular_auprc")
            .head(top_n)["graph_common_name"]
            .tolist()
        )
    return list(dict.fromkeys(names))


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


def add_strata(nodes: pd.DataFrame, grid_size_m: float, blocks_per_dim: int) -> pd.DataFrame:
    nodes = nodes.copy()
    nodes["spatial_block"] = assign_spatial_blocks(nodes, blocks_per_dim)
    cell_x = np.floor(nodes["x"].to_numpy(dtype=np.float64) / grid_size_m).astype(np.int64)
    cell_y = np.floor(nodes["y"].to_numpy(dtype=np.float64) / grid_size_m).astype(np.int64)
    nodes["spatial_cell"] = pd.Series(cell_x.astype(str) + "_" + cell_y.astype(str)).to_numpy()
    nodes["duration_bin"] = pd.cut(
        nodes["duration_minutes"].astype(float),
        bins=[0, 10, 30, 60, 120, np.inf],
        labels=["1-10", "11-30", "31-60", "61-120", "121+"],
        include_lowest=True,
    ).astype(str)
    nodes["distance_bin"] = pd.cut(
        nodes["effort_distance_km"].fillna(0.0).astype(float),
        bins=[-0.001, 0, 0.5, 2, 5, np.inf],
        labels=["0", "(0,0.5]", "(0.5,2]", "(2,5]", "5+"],
        include_lowest=True,
    ).astype(str)
    nodes["observer_bin"] = pd.cut(
        nodes["number_observers"].astype(float),
        bins=[0, 1, 2, np.inf],
        labels=["1", "2", "3+"],
        include_lowest=True,
    ).astype(str)
    return nodes


def load_graph_tables(graph_dir: Path, grid_size_m: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    blocks_per_dim = int(metadata.get("split", {}).get("spatial_blocks_per_dim", 8))
    checklist_columns = [
        "checklist_index",
        "protocol_name",
        "duration_minutes",
        "effort_distance_km",
        "number_observers",
        "county",
        "locality_id",
        "x",
        "y",
        "train_mask",
        "test_mask",
        *[col for col in ECOLOGY_COLUMNS],
    ]
    nodes = pd.read_parquet(
        graph_dir / "checklist_nodes.parquet",
        columns=checklist_columns,
    )
    nodes = add_strata(nodes, grid_size_m, blocks_per_dim)
    species = pd.read_csv(graph_dir / "species_nodes.csv")
    edges = pd.read_parquet(
        graph_dir / "positive_edges.parquet",
        columns=["checklist_index", "species_index", "split"],
    )
    return nodes, species, edges, metadata


def summarize_species(
    nodes: pd.DataFrame,
    species: pd.DataFrame,
    edges: pd.DataFrame,
    focus_species: list[str],
) -> pd.DataFrame:
    name_to_index = dict(zip(species["common_name"], species["species_index"]))
    rows = []
    for common_name in focus_species:
        species_index = name_to_index.get(common_name)
        if species_index is None:
            continue
        species_edges = edges.loc[edges["species_index"] == species_index]
        positive_by_split = {
            split: set(frame["checklist_index"].astype(int).tolist())
            for split, frame in species_edges.groupby("split")
        }
        for split_name, mask_column in [("train", "train_mask"), ("test", "test_mask")]:
            split_nodes = nodes.loc[nodes[mask_column]].copy()
            positives = split_nodes["checklist_index"].isin(
                positive_by_split.get(split_name, set())
            )
            positive_nodes = split_nodes.loc[positives]
            rows.append(
                {
                    "common_name": common_name,
                    "species_index": int(species_index),
                    "split": split_name,
                    "checklists": int(len(split_nodes)),
                    "positive_checklists": int(positives.sum()),
                    "prevalence": float(positives.mean()),
                    "positive_spatial_blocks": int(positive_nodes["spatial_block"].nunique()),
                    "positive_spatial_cells": int(positive_nodes["spatial_cell"].nunique()),
                    "positive_counties": int(positive_nodes["county"].nunique()),
                    "positive_localities": int(positive_nodes["locality_id"].nunique()),
                    "mean_positive_duration_minutes": float(
                        positive_nodes["duration_minutes"].mean()
                    )
                    if len(positive_nodes)
                    else np.nan,
                    "mean_positive_effort_distance_km": float(
                        positive_nodes["effort_distance_km"].astype(float).mean()
                    )
                    if len(positive_nodes)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_strata(
    nodes: pd.DataFrame,
    species: pd.DataFrame,
    edges: pd.DataFrame,
    focus_species: list[str],
) -> pd.DataFrame:
    name_to_index = dict(zip(species["common_name"], species["species_index"]))
    rows = []
    strata = [
        ("protocol", "protocol_name"),
        ("duration", "duration_bin"),
        ("distance", "distance_bin"),
        ("observers", "observer_bin"),
        ("spatial_block", "spatial_block"),
    ]
    test_nodes = nodes.loc[nodes["test_mask"]].copy()
    for common_name in focus_species:
        species_index = name_to_index.get(common_name)
        if species_index is None:
            continue
        positives = set(
            edges.loc[
                (edges["species_index"] == species_index) & (edges["split"] == "test"),
                "checklist_index",
            ]
            .astype(int)
            .tolist()
        )
        test_nodes["is_positive"] = test_nodes["checklist_index"].isin(positives)
        for stratum_type, column in strata:
            grouped = (
                test_nodes.groupby(column, dropna=False)["is_positive"]
                .agg(["count", "sum", "mean"])
                .reset_index()
            )
            for _, row in grouped.iterrows():
                rows.append(
                    {
                        "common_name": common_name,
                        "species_index": int(species_index),
                        "stratum_type": stratum_type,
                        "stratum": str(row[column]),
                        "checklists": int(row["count"]),
                        "positive_checklists": int(row["sum"]),
                        "prevalence": float(row["mean"]),
                    }
                )
    return pd.DataFrame(rows)


def summarize_covariates(
    nodes: pd.DataFrame,
    species: pd.DataFrame,
    edges: pd.DataFrame,
    focus_species: list[str],
) -> pd.DataFrame:
    name_to_index = dict(zip(species["common_name"], species["species_index"]))
    test_nodes = nodes.loc[nodes["test_mask"]].copy()
    rows = []
    for common_name in focus_species:
        species_index = name_to_index.get(common_name)
        if species_index is None:
            continue
        positives = set(
            edges.loc[
                (edges["species_index"] == species_index) & (edges["split"] == "test"),
                "checklist_index",
            ]
            .astype(int)
            .tolist()
        )
        positive_nodes = test_nodes.loc[test_nodes["checklist_index"].isin(positives)]
        for column in ECOLOGY_COLUMNS:
            rows.append(
                {
                    "common_name": common_name,
                    "species_index": int(species_index),
                    "covariate": column,
                    "test_all_mean": float(test_nodes[column].mean()),
                    "test_positive_mean": float(positive_nodes[column].mean())
                    if len(positive_nodes)
                    else np.nan,
                    "positive_minus_all": float(
                        positive_nodes[column].mean() - test_nodes[column].mean()
                    )
                    if len(positive_nodes)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_auprc_delta(comparison: pd.DataFrame, focus_species: list[str], output: Path) -> None:
    frame = comparison.loc[comparison["graph_common_name"].isin(focus_species)].copy()
    if frame.empty:
        return
    pivot = frame.pivot_table(
        index="graph_common_name",
        columns="model_label",
        values="graph_minus_tabular_auprc",
        aggfunc="first",
    )
    order = pivot.mean(axis=1).sort_values().index
    pivot = pivot.loc[order]
    height = max(5, 0.35 * len(pivot))
    ax = pivot.plot(kind="barh", figsize=(10, height), width=0.75)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Graph minus tabular AUPRC")
    ax.set_ylabel("")
    ax.set_title("Spatial GNN species AUPRC deltas")
    ax.legend(title="Model", loc="best")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_coverage(summary: pd.DataFrame, output: Path) -> None:
    frame = summary.loc[summary["split"] == "test"].copy()
    if frame.empty:
        return
    frame = frame.sort_values("positive_checklists")
    fig, axes = plt.subplots(1, 2, figsize=(12, max(5, 0.35 * len(frame))), sharey=True)
    axes[0].barh(frame["common_name"], frame["positive_checklists"], color="#4C78A8")
    axes[0].set_xlabel("Held-out positive checklists")
    axes[0].set_title("Test positives")
    axes[1].barh(frame["common_name"], frame["positive_spatial_cells"], color="#F58518")
    axes[1].set_xlabel("Held-out positive spatial cells")
    axes[1].set_title("Spatial coverage")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def plot_species_maps(
    nodes: pd.DataFrame,
    species: pd.DataFrame,
    edges: pd.DataFrame,
    focus_species: list[str],
    output_dir: Path,
    boundary: gpd.GeoDataFrame | None,
) -> None:
    name_to_index = dict(zip(species["common_name"], species["species_index"]))
    test_nodes = nodes.loc[nodes["test_mask"]]
    for common_name in focus_species:
        species_index = name_to_index.get(common_name)
        if species_index is None:
            continue
        positives = set(
            edges.loc[
                (edges["species_index"] == species_index) & (edges["split"] == "test"),
                "checklist_index",
            ]
            .astype(int)
            .tolist()
        )
        positive_nodes = test_nodes.loc[test_nodes["checklist_index"].isin(positives)]
        if positive_nodes.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(test_nodes["x"], test_nodes["y"], s=1, c="#D0D0D0", alpha=0.25, linewidths=0)
        ax.scatter(positive_nodes["x"], positive_nodes["y"], s=3, c="#D62728", alpha=0.7, linewidths=0)
        draw_boundary(ax, boundary)
        ax.set_title(f"{common_name}: held-out positives")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        safe_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in common_name).strip("_")
        plt.tight_layout()
        plt.savefig(output_dir / f"{safe_name}_test_positive_map.png", dpi=180)
        plt.close(fig)


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


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else graph_dir / "spatial_gnn_baselines" / "diagnostics" / "species_diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_paths = [Path(path) for path in args.comparison_csv]
    if not comparison_paths:
        comparison_paths = discover_comparison_csvs(graph_dir)
    comparison = load_comparisons(comparison_paths)
    focus_species = choose_focus_species(
        comparison, args.focus_species, args.top_delta_species
    )

    nodes, species, edges, metadata = load_graph_tables(graph_dir, args.spatial_grid_size_m)
    summary = summarize_species(nodes, species, edges, focus_species)
    strata = summarize_strata(nodes, species, edges, focus_species)
    covariates = summarize_covariates(nodes, species, edges, focus_species)

    focus_comparison = comparison.loc[
        comparison["graph_common_name"].isin(focus_species)
    ].copy()
    focus_comparison.to_csv(output_dir / "focus_species_model_deltas.csv", index=False)
    summary.to_csv(output_dir / "focus_species_split_summary.csv", index=False)
    strata.to_csv(output_dir / "focus_species_test_strata.csv", index=False)
    covariates.to_csv(output_dir / "focus_species_test_covariates.csv", index=False)
    (output_dir / "diagnostic_metadata.json").write_text(
        json.dumps(
            {
                "graph_dir": str(graph_dir),
                "comparison_csvs": [str(path) for path in comparison_paths],
                "focus_species": focus_species,
                "spatial_grid_size_m": args.spatial_grid_size_m,
                "split": metadata.get("split", {}),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_auprc_delta(
        comparison,
        focus_species,
        output_dir / "focus_species_auprc_delta.png",
    )
    plot_coverage(summary, output_dir / "focus_species_test_coverage.png")
    boundary = load_boundary(args.boundary, args.map_crs)
    plot_species_maps(nodes, species, edges, focus_species, output_dir, boundary)

    print(f"Wrote species diagnostics to {output_dir}")
    print("\nFocus species:")
    for name in focus_species:
        print(f"  {name}")
    print("\nTest split summary:")
    print(
        summary.loc[summary["split"] == "test"]
        .sort_values("positive_checklists")
        [
            [
                "common_name",
                "positive_checklists",
                "prevalence",
                "positive_spatial_blocks",
                "positive_spatial_cells",
                "positive_counties",
            ]
        ]
        .to_string(index=False, float_format="%.4f")
    )


if __name__ == "__main__":
    main()
