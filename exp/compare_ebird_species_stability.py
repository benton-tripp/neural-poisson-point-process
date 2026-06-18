"""
Compare graph-vs-tabular species deltas across two validation splits.

This is intended for checking whether GNN gains/losses are stable across split
definitions, or whether they are artifacts of one held-out geography.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PRIMARY = (
    "data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/"
    "residual_primary_graph_vs_tabular_species.csv"
)
DEFAULT_COMPARISON = (
    "data/ebird/graph_top100_spatial_10x10_coastalstress/spatial_gnn_baselines/"
    "spatial_gcn_frozen_access_h64_l2_z64_graph_vs_tabular_species.csv"
)
DEFAULT_OUTPUT = (
    "data/ebird/graph_top100_spatial_10x10_coastalstress/spatial_gnn_baselines/"
    "diagnostics/species_stability/"
    "primary_vs_coastalstress_species_stability.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare species-level graph-vs-tabular deltas across two splits."
    )
    parser.add_argument(
        "--primary",
        default=DEFAULT_PRIMARY,
        help=f"Primary graph-vs-tabular species CSV. Defaults to {DEFAULT_PRIMARY}.",
    )
    parser.add_argument(
        "--comparison",
        default=DEFAULT_COMPARISON,
        help=(
            "Comparison graph-vs-tabular species CSV. Defaults to the "
            f"coastal-stress file: {DEFAULT_COMPARISON}."
        ),
    )
    parser.add_argument(
        "--primary-label",
        default="primary",
        help="Label for the primary split. Defaults to primary.",
    )
    parser.add_argument(
        "--comparison-label",
        default="coastalstress",
        help="Label for the comparison split. Defaults to coastalstress.",
    )
    parser.add_argument(
        "--metric",
        choices=["auprc", "auroc"],
        default="auprc",
        help="Graph-minus-tabular metric used for stability labels. Defaults to auprc.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help=(
            "Minimum graph-minus-tabular delta treated as a meaningful gain/loss. "
            "Defaults to 0.005."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Defaults to {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args()


def load_metrics(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing species comparison CSV: {path}")
    frame = pd.read_csv(path)
    required = {
        "species_key",
        "graph_common_name",
        "graph_minus_tabular_auprc",
        "graph_minus_tabular_auroc",
        "graph_auprc",
        "tabular_auprc",
        "graph_auroc",
        "tabular_auroc",
        "graph_calibration_error",
        "tabular_calibration_error",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    keep = [
        "species_key",
        "graph_common_name",
        "graph_minus_tabular_auprc",
        "graph_minus_tabular_auroc",
        "graph_auprc",
        "tabular_auprc",
        "graph_auroc",
        "tabular_auroc",
        "graph_calibration_error",
        "tabular_calibration_error",
    ]
    optional = ["graph_observed_rate", "tabular_test_prevalence", "graph_pairs"]
    keep.extend(column for column in optional if column in frame.columns)
    frame = frame[keep].copy()
    rename = {
        column: f"{label}_{column}"
        for column in frame.columns
        if column not in {"species_key", "graph_common_name"}
    }
    return frame.rename(
        columns={
            "graph_common_name": "common_name",
            **rename,
        }
    )


def classify(primary_delta: float, comparison_delta: float, threshold: float) -> str:
    primary_gain = primary_delta >= threshold
    comparison_gain = comparison_delta >= threshold
    primary_loss = primary_delta <= -threshold
    comparison_loss = comparison_delta <= -threshold

    if primary_gain and comparison_gain:
        return "consistently_helped"
    if primary_loss and comparison_loss:
        return "consistently_hurt"
    if primary_gain and comparison_loss:
        return "primary_helped_comparison_hurt"
    if primary_loss and comparison_gain:
        return "primary_hurt_comparison_helped"
    if primary_gain:
        return "primary_only_helped"
    if comparison_gain:
        return "comparison_only_helped"
    if primary_loss:
        return "primary_only_hurt"
    if comparison_loss:
        return "comparison_only_hurt"
    return "neutral_or_small"


def print_table(title: str, frame: pd.DataFrame, columns: list[str], n: int = 12) -> None:
    print(f"\n{title}:")
    if frame.empty:
        print("(none)")
        return
    print(frame[columns].head(n).to_string(index=False, float_format="%.4f"))


def main() -> None:
    args = parse_args()
    primary_path = Path(args.primary)
    comparison_path = Path(args.comparison)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    primary = load_metrics(primary_path, args.primary_label)
    comparison = load_metrics(comparison_path, args.comparison_label)
    merged = primary.merge(
        comparison,
        on="species_key",
        suffixes=(f"_{args.primary_label}", f"_{args.comparison_label}"),
        how="inner",
        validate="one_to_one",
    )
    if "common_name_primary" in merged.columns:
        merged["common_name"] = merged["common_name_primary"]
        merged = merged.drop(
            columns=[
                column
                for column in ["common_name_primary", "common_name_comparison"]
                if column in merged.columns
            ]
        )

    primary_delta = f"{args.primary_label}_graph_minus_tabular_{args.metric}"
    comparison_delta = f"{args.comparison_label}_graph_minus_tabular_{args.metric}"
    primary_calibration_delta = (
        f"{args.primary_label}_graph_calibration_error"
    )
    comparison_calibration_delta = (
        f"{args.comparison_label}_graph_calibration_error"
    )
    if primary_delta not in merged.columns or comparison_delta not in merged.columns:
        raise ValueError(
            f"Could not find delta columns {primary_delta} and {comparison_delta}."
        )

    merged["stability_class"] = [
        classify(primary_value, comparison_value, args.threshold)
        for primary_value, comparison_value in zip(
            merged[primary_delta], merged[comparison_delta]
        )
    ]
    merged[f"mean_graph_minus_tabular_{args.metric}"] = merged[
        [primary_delta, comparison_delta]
    ].mean(axis=1)
    merged[f"min_graph_minus_tabular_{args.metric}"] = merged[
        [primary_delta, comparison_delta]
    ].min(axis=1)
    merged[f"max_graph_minus_tabular_{args.metric}"] = merged[
        [primary_delta, comparison_delta]
    ].max(axis=1)
    merged[f"{args.comparison_label}_minus_{args.primary_label}_{args.metric}"] = (
        merged[comparison_delta] - merged[primary_delta]
    )
    merged[f"abs_{args.comparison_label}_minus_{args.primary_label}_{args.metric}"] = (
        merged[f"{args.comparison_label}_minus_{args.primary_label}_{args.metric}"].abs()
    )
    if primary_calibration_delta in merged.columns and comparison_calibration_delta in merged.columns:
        merged[
            f"{args.comparison_label}_minus_{args.primary_label}_graph_calibration_error"
        ] = merged[comparison_calibration_delta] - merged[primary_calibration_delta]

    sort_column = f"mean_graph_minus_tabular_{args.metric}"
    merged = merged.sort_values(sort_column, ascending=False)
    merged.to_csv(output_path, index=False)

    summary = {
        "primary": str(primary_path),
        "comparison": str(comparison_path),
        "metric": args.metric,
        "threshold": args.threshold,
        "species_joined": int(len(merged)),
        "class_counts": {
            key: int(value)
            for key, value in merged["stability_class"].value_counts().sort_index().items()
        },
        "mean_primary_delta": float(merged[primary_delta].mean()),
        "mean_comparison_delta": float(merged[comparison_delta].mean()),
        "median_primary_delta": float(merged[primary_delta].median()),
        "median_comparison_delta": float(merged[comparison_delta].median()),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    columns = [
        "common_name",
        "stability_class",
        primary_delta,
        comparison_delta,
        f"mean_graph_minus_tabular_{args.metric}",
        f"{args.comparison_label}_minus_{args.primary_label}_{args.metric}",
    ]
    print("\nSpecies stability summary:")
    print(json.dumps(summary, indent=2))
    print_table(
        "Consistently helped species",
        merged[merged["stability_class"] == "consistently_helped"].sort_values(
            f"min_graph_minus_tabular_{args.metric}", ascending=False
        ),
        columns,
    )
    print_table(
        "Consistently hurt species",
        merged[merged["stability_class"] == "consistently_hurt"].sort_values(
            f"max_graph_minus_tabular_{args.metric}"
        ),
        columns,
    )
    print_table(
        "Largest split-sensitive changes",
        merged.sort_values(
            f"abs_{args.comparison_label}_minus_{args.primary_label}_{args.metric}",
            ascending=False,
        ),
        columns,
    )
    print(f"\nWrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
