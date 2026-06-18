"""
Compare calibration outputs from eBird tabular baselines.

Run from the project root after baseline runs have produced *_calibration.csv:

    python exp/compare_ebird_calibration.py --top-species 100 --split spatial-stratified
"""

from __future__ import annotations

import argparse
import json
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


def summary_path(
    baseline_dir: Path,
    top_species: int,
    feature_set: str,
    split: str,
    model: str,
) -> Path:
    return (
        baseline_dir
        / f"top{top_species}_{feature_set}{model_file_suffix(model)}{split_suffix(split)}_summary.json"
    )


def split_signature(path: Path) -> tuple | None:
    if not path.exists():
        return None
    summary = json.loads(path.read_text(encoding="utf-8"))
    split = summary.get("split", {})
    return (
        split.get("split"),
        split.get("spatial_blocks_per_dim"),
        tuple(split.get("test_blocks_ids", [])),
        split.get("test_fraction_actual"),
        split.get("mean_absolute_standardized_balance_error"),
    )


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
    candidate_feature_sets = ["effort", "ecology", "both", "both-regime"]
    signatures = {
        feature_set: split_signature(
            summary_path(
                baseline_dir, args.top_species, feature_set, args.split, args.model
            )
        )
        for feature_set in candidate_feature_sets
    }
    reference_feature = "both" if signatures.get("both") is not None else None
    if reference_feature is None:
        reference_feature = next(
            (name for name in candidate_feature_sets if signatures.get(name) is not None),
            None,
        )
    if reference_feature is None:
        raise FileNotFoundError("No matching summary JSON files were found.")
    reference_signature = signatures[reference_feature]

    frames = []
    skipped = []
    for feature_set in candidate_feature_sets:
        signature = signatures.get(feature_set)
        if signature is None:
            if feature_set == "both-regime":
                continue
            raise FileNotFoundError(
                f"Missing baseline summary file for feature set: {feature_set}"
            )
        if signature != reference_signature:
            skipped.append(feature_set)
            continue
        try:
            frames.append(
                read_calibration(
                    baseline_dir, args.top_species, feature_set, args.split, args.model
                )
            )
        except FileNotFoundError:
            if feature_set == "both-regime":
                continue
            raise
    if len(frames) < 2:
        raise ValueError("Need at least two compatible calibration files to compare.")
    calibration = pd.concat(frames, ignore_index=True)

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
    if skipped:
        print(
            "Skipped incompatible split configuration for feature set(s): "
            + ", ".join(skipped)
        )
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
