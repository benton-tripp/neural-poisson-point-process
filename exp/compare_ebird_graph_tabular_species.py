"""
Compare graph link baseline species diagnostics against tabular MLP metrics.

Run from the project root after running the graph link baseline:

    python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial"
DEFAULT_BASELINE_DIR = "data/ebird/baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare graph link species metrics with tabular species metrics."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--baseline-dir",
        default=DEFAULT_BASELINE_DIR,
        help=f"Tabular baseline directory. Defaults to {DEFAULT_BASELINE_DIR}.",
    )
    parser.add_argument(
        "--top-species",
        type=int,
        default=100,
        help="Top-N species prefix for tabular metrics. Defaults to 100.",
    )
    parser.add_argument(
        "--tabular-model",
        choices=["linear", "mlp"],
        default="mlp",
        help="Tabular model family to compare. Defaults to mlp.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["effort", "ecology", "both"],
        default="both",
        help="Tabular feature set to compare. Defaults to both.",
    )
    parser.add_argument(
        "--split",
        choices=["temporal", "spatial-stratified"],
        default="spatial-stratified",
        help="Tabular split suffix to compare. Defaults to spatial-stratified.",
    )
    parser.add_argument(
        "--link-output-dir",
        default=None,
        help="Graph link output directory. Defaults to graph-dir/link_baselines.",
    )
    parser.add_argument(
        "--graph-species-metrics",
        default=None,
        help=(
            "Optional graph species metrics CSV. Defaults to the all-pairs "
            "metrics file when present, otherwise sampled-edge metrics."
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["auroc", "auprc", "calibration_error"],
        default="auprc",
        help="Metric used for sorting graph-minus-tabular differences. Defaults to auprc.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output CSV path. Defaults to link output directory.",
    )
    return parser.parse_args()


def split_suffix(split: str) -> str:
    return "" if split == "temporal" else f"_{split}"


def model_file_suffix(model: str) -> str:
    return "" if model == "linear" else f"_{model}"


def load_tabular_metrics(
    baseline_dir: Path,
    top_species: int,
    feature_set: str,
    split: str,
    model: str,
) -> pd.DataFrame:
    path = (
        baseline_dir
        / f"top{top_species}_{feature_set}{model_file_suffix(model)}{split_suffix(split)}_metrics.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing tabular metrics file: {path}")
    model_name = f"{model}_{feature_set}"
    frame = pd.read_csv(path)
    frame = frame[frame["model"] == model_name].copy()
    if frame.empty:
        raise ValueError(f"No rows for model {model_name} in {path}")
    keep = [
        "species_key",
        "common_name",
        "test_prevalence",
        "test_detections",
        "auroc",
        "auprc",
    ]
    optional = ["mean_predicted", "calibration_error"]
    keep.extend([column for column in optional if column in frame.columns])
    frame = frame[keep].copy()
    if "mean_predicted" not in frame.columns:
        frame["mean_predicted"] = np.nan
    if "calibration_error" not in frame.columns:
        frame["calibration_error"] = np.nan
    return frame.rename(
        columns={
            "common_name": "tabular_common_name",
            "test_prevalence": "tabular_test_prevalence",
            "test_detections": "tabular_test_detections",
            "mean_predicted": "tabular_mean_predicted",
            "calibration_error": "tabular_calibration_error",
            "auroc": "tabular_auroc",
            "auprc": "tabular_auprc",
        }
    )


def load_graph_metrics(
    graph_dir: Path,
    link_output_dir: Path,
    graph_species_metrics: Path | None,
) -> tuple[pd.DataFrame, Path, str]:
    species_path = graph_dir / "species_nodes.csv"
    if graph_species_metrics:
        graph_path = graph_species_metrics
    else:
        corrected_all_pairs_path = (
            link_output_dir
            / "species_embedding_link_test_all_pairs_prior_corrected_species_metrics.csv"
        )
        all_pairs_path = (
            link_output_dir / "species_embedding_link_test_all_pairs_species_metrics.csv"
        )
        sampled_path = link_output_dir / "species_embedding_link_test_species_metrics.csv"
        if corrected_all_pairs_path.exists():
            graph_path = corrected_all_pairs_path
        elif all_pairs_path.exists():
            graph_path = all_pairs_path
        else:
            graph_path = sampled_path
    if not species_path.exists():
        raise FileNotFoundError(f"Missing species node file: {species_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph species metrics file: {graph_path}")

    species = pd.read_csv(species_path)[
        ["species_index", "species_key", "taxon_concept_id"]
    ]
    graph = pd.read_csv(graph_path)
    if "species_key" not in graph.columns:
        graph = graph.merge(
            species, on="species_index", how="left", validate="one_to_one"
        )
    if graph["species_key"].isna().any():
        missing = int(graph["species_key"].isna().sum())
        raise ValueError(f"Could not map {missing} graph species rows to species keys.")
    count_column = "pairs" if "pairs" in graph.columns else "edges"
    positive_column = "positives"
    negative_column = "negatives"
    if count_column == "pairs" and "all_species_link" in graph_path.name:
        target = "all_species_all_pairs"
    elif count_column == "pairs" and "prior_corrected" in graph_path.name:
        target = "prior_corrected_all_pairs"
    elif count_column == "pairs":
        target = "all_pairs"
    else:
        target = "sampled_edges"
    return graph.rename(
        columns={
            "common_name": "graph_common_name",
            "scientific_name": "graph_scientific_name",
            count_column: "graph_pairs",
            positive_column: "graph_positives",
            negative_column: "graph_negatives",
            "observed_rate": "graph_observed_rate",
            "mean_predicted": "graph_mean_predicted",
            "calibration_error": "graph_calibration_error",
            "auroc": "graph_auroc",
            "auprc": "graph_auprc",
        }
    ), graph_path, target


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    baseline_dir = Path(args.baseline_dir)
    link_output_dir = (
        Path(args.link_output_dir) if args.link_output_dir else graph_dir / "link_baselines"
    )

    graph, graph_path, graph_target = load_graph_metrics(
        graph_dir,
        link_output_dir,
        Path(args.graph_species_metrics) if args.graph_species_metrics else None,
    )
    tabular = load_tabular_metrics(
        baseline_dir,
        args.top_species,
        args.feature_set,
        args.split,
        args.tabular_model,
    )
    comparison = graph.merge(tabular, on="species_key", how="inner", validate="one_to_one")
    if len(comparison) != min(len(graph), len(tabular)):
        print(
            f"Warning: joined {len(comparison)} rows from "
            f"{len(graph)} graph rows and {len(tabular)} tabular rows."
        )

    comparison["graph_minus_tabular_auroc"] = (
        comparison["graph_auroc"] - comparison["tabular_auroc"]
    )
    comparison["graph_minus_tabular_auprc"] = (
        comparison["graph_auprc"] - comparison["tabular_auprc"]
    )
    comparison["graph_minus_tabular_prevalence"] = (
        comparison["graph_observed_rate"]
        - comparison["tabular_test_prevalence"]
    )
    comparison["name_match"] = (
        comparison["graph_common_name"] == comparison["tabular_common_name"]
    )

    output = (
        Path(args.output)
        if args.output
        else (Path(args.graph_species_metrics).parent if args.graph_species_metrics else link_output_dir)
        / (
            f"top{args.top_species}_{args.tabular_model}_{args.feature_set}_"
            f"{args.split}_{graph_path.stem}_graph_vs_tabular_species.csv"
        )
    )
    comparison = comparison.sort_values(
        f"graph_minus_tabular_{args.metric}"
        if args.metric != "calibration_error"
        else "graph_calibration_error",
        ascending=args.metric == "calibration_error",
    )
    comparison.to_csv(output, index=False)

    metric_columns = [
        "graph_common_name",
        "tabular_test_prevalence",
        "graph_observed_rate",
        "tabular_auroc",
        "graph_auroc",
        "graph_minus_tabular_auroc",
        "tabular_auprc",
        "graph_auprc",
        "graph_minus_tabular_auprc",
        "tabular_calibration_error",
        "graph_calibration_error",
    ]
    print("\nLargest graph AUROC gains over tabular:")
    print(
        comparison.sort_values("graph_minus_tabular_auroc", ascending=False)[
            metric_columns
        ]
        .head(12)
        .to_string(index=False, float_format="%.4f")
    )
    print("\nLargest graph AUROC losses vs tabular:")
    print(
        comparison.sort_values("graph_minus_tabular_auroc")[metric_columns]
        .head(12)
        .to_string(index=False, float_format="%.4f")
    )
    auprc_label = "all-pairs" if "all_pairs" in graph_target else "sampled-edge"
    print(f"\nLargest graph {auprc_label} AUPRC gains over tabular:")
    print(
        comparison.sort_values("graph_minus_tabular_auprc", ascending=False)[
            metric_columns
        ]
        .head(12)
        .to_string(index=False, float_format="%.4f")
    )
    print(f"\nSmallest graph {auprc_label} AUPRC gains over tabular:")
    print(
        comparison.sort_values("graph_minus_tabular_auprc")[metric_columns]
        .head(12)
        .to_string(index=False, float_format="%.4f")
    )
    print("\nLargest graph species calibration errors:")
    print(
        comparison.sort_values("graph_calibration_error", ascending=False)[
            metric_columns
        ]
        .head(12)
        .to_string(index=False, float_format="%.4f")
    )
    print(
        f"\nGraph metrics source: {graph_path} ({graph_target})."
    )
    if graph_target == "sampled_edges":
        print(
            "Note: graph AUPRC uses the sampled edge distribution, while tabular "
            "AUPRC uses all held-out checklist/species pairs. Use AUROC and "
            "calibration for the cleaner cross-framework comparison until the "
            "graph evaluation is run on the same all-pairs target distribution."
        )
    else:
        print(
            "Note: graph and tabular metrics are now evaluated on the same "
            "all-pairs target distribution, so AUPRC is comparable."
        )
    print(f"\nWrote {output}")

    if not comparison["name_match"].all():
        mismatches = comparison.loc[
            ~comparison["name_match"], ["species_key", "graph_common_name", "tabular_common_name"]
        ]
        print("\nName mismatches:")
        print(mismatches.to_string(index=False))


if __name__ == "__main__":
    main()
