"""
Compare per-species eBird tabular baseline metrics.

Run from the project root after running effort, ecology, and both baselines:

    python exp/compare_ebird_tabular_baselines.py --baseline-dir data/ebird/baselines --top-species 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BASELINE_DIR = "data/ebird/baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare per-species metrics from eBird tabular baselines."
    )
    parser.add_argument(
        "--baseline-dir",
        default=DEFAULT_BASELINE_DIR,
        help=f"Directory with topN_*_metrics.csv files. Defaults to {DEFAULT_BASELINE_DIR}.",
    )
    parser.add_argument(
        "--top-species",
        type=int,
        default=20,
        help="Top-N species prefix to compare. Defaults to 20.",
    )
    parser.add_argument(
        "--split",
        choices=["temporal", "spatial-stratified"],
        default="temporal",
        help="Split suffix to compare. Defaults to temporal.",
    )
    parser.add_argument(
        "--model",
        choices=["linear", "mlp"],
        default="linear",
        help="Model family to compare. Defaults to linear.",
    )
    parser.add_argument(
        "--metric",
        choices=["auroc", "auprc"],
        default="auprc",
        help="Metric used for sorting species differences. Defaults to auprc.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path. Defaults to baseline-dir/topN_model_comparison.csv.",
    )
    return parser.parse_args()


def split_suffix(split: str) -> str:
    return "" if split == "temporal" else f"_{split}"


def model_file_suffix(model: str) -> str:
    return "" if model == "linear" else f"_{model}"


def read_model_metrics(
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
        raise FileNotFoundError(f"Missing baseline metrics file: {path}")
    frame = pd.read_csv(path)
    model_name = f"{model}_{feature_set}"
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
    frame = frame[keep]
    return frame.rename(
        columns={
            "auroc": f"{feature_set}_auroc",
            "auprc": f"{feature_set}_auprc",
        }
    )


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    comparison = read_model_metrics(
        baseline_dir, args.top_species, "effort", args.split, args.model
    )
    for feature_set in ["ecology", "both"]:
        comparison = comparison.merge(
            read_model_metrics(
                baseline_dir, args.top_species, feature_set, args.split, args.model
            ),
            on=["species_key", "common_name", "test_prevalence", "test_detections"],
            how="inner",
        )

    for metric in ["auroc", "auprc"]:
        comparison[f"both_minus_effort_{metric}"] = (
            comparison[f"both_{metric}"] - comparison[f"effort_{metric}"]
        )
        comparison[f"both_minus_ecology_{metric}"] = (
            comparison[f"both_{metric}"] - comparison[f"ecology_{metric}"]
        )
        comparison[f"effort_minus_ecology_{metric}"] = (
            comparison[f"effort_{metric}"] - comparison[f"ecology_{metric}"]
        )

    sort_column = f"both_minus_effort_{args.metric}"
    comparison = comparison.sort_values(sort_column, ascending=False)

    output = (
        Path(args.output)
        if args.output
        else baseline_dir
        / f"top{args.top_species}{model_file_suffix(args.model)}{split_suffix(args.split)}_model_comparison.csv"
    )
    comparison.to_csv(output, index=False)

    display_columns = [
        "common_name",
        "test_prevalence",
        "test_detections",
        "effort_auprc",
        "ecology_auprc",
        "both_auprc",
        "both_minus_effort_auprc",
        "both_minus_ecology_auprc",
    ]
    print(comparison[display_columns].to_string(index=False, float_format="%.4f"))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
