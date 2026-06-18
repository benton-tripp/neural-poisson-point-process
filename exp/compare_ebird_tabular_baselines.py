"""
Compare per-species eBird tabular baseline metrics.

Run from the project root after running effort, ecology, and both baselines:

    python exp/compare_ebird_tabular_baselines.py --baseline-dir data/ebird/baselines --top-species 20
"""

from __future__ import annotations

import argparse
import json
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


def metrics_path(
    baseline_dir: Path,
    top_species: int,
    feature_set: str,
    split: str,
    model: str,
) -> Path:
    return (
        baseline_dir
        / f"top{top_species}_{feature_set}{model_file_suffix(model)}{split_suffix(split)}_metrics.csv"
    )


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


def read_model_metrics(
    baseline_dir: Path,
    top_species: int,
    feature_set: str,
    split: str,
    model: str,
) -> pd.DataFrame:
    path = metrics_path(baseline_dir, top_species, feature_set, split, model)
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
            "test_prevalence": f"{feature_set}_test_prevalence",
            "test_detections": f"{feature_set}_test_detections",
            "auroc": f"{feature_set}_auroc",
            "auprc": f"{feature_set}_auprc",
        }
    )


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

    comparison = None
    loaded_feature_sets = []
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
            feature_metrics = read_model_metrics(
                baseline_dir, args.top_species, feature_set, args.split, args.model
            )
        except FileNotFoundError:
            if feature_set == "both-regime":
                continue
            raise
        if comparison is None:
            comparison = feature_metrics
        else:
            comparison = comparison.merge(
                feature_metrics,
                on=["species_key", "common_name"],
                how="inner",
            )
        loaded_feature_sets.append(feature_set)
    if comparison is None or len(loaded_feature_sets) < 2:
        raise ValueError("Need at least two compatible feature sets to compare.")

    for metric in ["auroc", "auprc"]:
        if {"both", "effort"}.issubset(loaded_feature_sets):
            comparison[f"both_minus_effort_{metric}"] = (
                comparison[f"both_{metric}"] - comparison[f"effort_{metric}"]
            )
        if {"both", "ecology"}.issubset(loaded_feature_sets):
            comparison[f"both_minus_ecology_{metric}"] = (
                comparison[f"both_{metric}"] - comparison[f"ecology_{metric}"]
            )
        if {"effort", "ecology"}.issubset(loaded_feature_sets):
            comparison[f"effort_minus_ecology_{metric}"] = (
                comparison[f"effort_{metric}"] - comparison[f"ecology_{metric}"]
            )
        if "both-regime" in loaded_feature_sets:
            comparison[f"both-regime_minus_both_{metric}"] = (
                comparison[f"both-regime_{metric}"] - comparison[f"both_{metric}"]
            )

    sort_column = (
        f"both-regime_minus_both_{args.metric}"
        if "both-regime" in loaded_feature_sets
        else f"both_minus_effort_{args.metric}"
    )
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
    ]
    prevalence_source = "both" if "both" in loaded_feature_sets else loaded_feature_sets[0]
    display_columns.extend(
        [
            f"{prevalence_source}_test_prevalence",
            f"{prevalence_source}_test_detections",
        ]
    )
    for feature_set in loaded_feature_sets:
        display_columns.append(f"{feature_set}_auprc")
    for column in [
        "both_minus_effort_auprc",
        "both_minus_ecology_auprc",
        "both-regime_minus_both_auprc",
    ]:
        if column in comparison.columns:
            display_columns.append(column)
    if "both-regime" in loaded_feature_sets:
        pass
    if skipped:
        print(
            "Skipped incompatible split configuration for feature set(s): "
            + ", ".join(skipped)
        )
    print(comparison[display_columns].to_string(index=False, float_format="%.4f"))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
