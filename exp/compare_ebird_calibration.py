"""
Compare calibration outputs from eBird tabular baselines.

Run from the project root after baseline runs have produced *_calibration.csv:

    python exp/compare_ebird_calibration.py --top-species 100 --split spatial-stratified
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_BASELINE_DIR = "data/ebird/baselines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare calibration rows from eBird tabular baselines."
    )
    parser.add_argument(
        "--baseline-dir",
        default=DEFAULT_BASELINE_DIR,
        help=f"Directory with topN_*_calibration.csv files. Defaults to {DEFAULT_BASELINE_DIR}.",
    )
    parser.add_argument(
        "--top-species",
        type=int,
        default=100,
        help="Top-N species prefix to compare. Defaults to 100.",
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
        "--output",
        default=None,
        help="Optional CSV path. Defaults to baseline-dir/topN_split_calibration_comparison.csv.",
    )
    return parser.parse_args()


def split_suffix(split: str) -> str:
    return "" if split == "temporal" else f"_{split}"


def model_file_suffix(model: str) -> str:
    return "" if model == "linear" else f"_{model}"


def read_calibration(
    baseline_dir: Path,
    top_species: int,
    feature_set: str,
    split: str,
    model: str,
):
    path = (
        baseline_dir
        / f"top{top_species}_{feature_set}{model_file_suffix(model)}{split_suffix(split)}_calibration.csv"
    )
    if not path.exists():
        raise FileNotFoundError(f"Missing calibration file: {path}")
    frame = pd.read_csv(path)
    model_name = f"{model}_{feature_set}"
    frame = frame[frame["model"] == model_name].copy()
    if frame.empty:
        raise ValueError(f"No rows for model {model_name} in {path}")
    frame["feature_set"] = feature_set
    return frame


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir)
    calibration = pd.concat(
        [
            read_calibration(
                baseline_dir, args.top_species, feature_set, args.split, args.model
            )
            for feature_set in ["effort", "ecology", "both"]
        ],
        ignore_index=True,
    )

    output = (
        Path(args.output)
        if args.output
        else baseline_dir
        / f"top{args.top_species}{model_file_suffix(args.model)}{split_suffix(args.split)}_calibration_comparison.csv"
    )
    calibration.sort_values(
        ["calibration_type", "stratum", "feature_set"]
    ).to_csv(output, index=False)

    effort = calibration[
        calibration["calibration_type"].str.startswith("effort_")
    ].copy()
    effort = effort.sort_values("calibration_error", ascending=False)
    probability = calibration[
        calibration["calibration_type"] == "predicted_probability_bin"
    ].copy()
    probability = probability.sort_values("calibration_error", ascending=False)

    display_cols = [
        "feature_set",
        "calibration_type",
        "stratum",
        "checklists",
        "mean_predicted",
        "observed_rate",
        "calibration_error",
    ]
    print("\nLargest effort-stratum calibration errors:")
    print(effort[display_cols].head(15).to_string(index=False, float_format="%.4f"))

    print("\nLargest predicted-probability-bin calibration errors:")
    print(
        probability[
            [
                "feature_set",
                "stratum",
                "pairs",
                "mean_predicted",
                "observed_rate",
                "calibration_error",
            ]
        ]
        .head(15)
        .to_string(index=False, float_format="%.4f")
    )

    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
