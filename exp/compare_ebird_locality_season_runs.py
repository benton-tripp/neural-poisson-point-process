"""
Compare saved locality-season/checklist detection runs.

The first run is treated as the reference. The script compares pooled metrics,
species metrics, focus-species monthly phenology, and binned environmental
responses without retraining any model.

Run from the project root:

    python exp/compare_ebird_locality_season_runs.py --run two_component_checklist_detection_e10_d20 --run two_component_checklist_detection_shrink_r0p01_a0p01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODEL_DIR = "data/ebird/locality_season_top100/detection_models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved locality-season detection model runs."
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Detection model output directory. Defaults to {DEFAULT_MODEL_DIR}.",
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help=(
            "Run name prefix. Repeat for each run; the first run is the "
            "reference."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to model-dir/diagnostics/run_comparisons.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Species rows to print for the largest AUPRC changes. Defaults to 10.",
    )
    parser.add_argument(
        "--comparison-name",
        default=None,
        help=(
            "Optional short output filename prefix. Defaults to a compact "
            "reference/run-count/hash identifier."
        ),
    )
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required output: {path}")
    return pd.read_csv(path)


def read_required_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required output: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(dtype=float), weights=weights))


def shape_spearman(observed: pd.Series, predicted: pd.Series) -> float:
    return float(
        observed.rank(method="average").corr(predicted.rank(method="average"))
    )


def safe_filename_stem(
    run_names: list[str],
    comparison_name: str | None,
) -> str:
    if comparison_name:
        stem = comparison_name
    else:
        digest = hashlib.sha1(
            "\n".join(run_names).encode("utf-8")
        ).hexdigest()[:10]
        stem = f"{run_names[0]}_vs_{len(run_names) - 1}runs_{digest}"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    if not stem:
        raise ValueError("Comparison name does not contain filename-safe characters.")
    return stem[:120]


def run_summary(model_dir: Path, run_name: str) -> dict:
    summary = read_required_json(model_dir / f"{run_name}_summary.json")
    model = summary["models"]["two_component"]
    species = summary["species_metric_means"]["two_component"]
    parameters = (
        summary.get("detection_parameters", {}).get("two_component", {})
    )
    regularization = summary.get("two_component_regularization", {})
    return {
        "run": run_name,
        "bce": model["bce"],
        "micro_auroc": model["micro_auroc"],
        "micro_auprc": model["micro_auprc"],
        "ece": model["ece"],
        "max_bin_error": model["max_bin_error"],
        "calibration_error": model["calibration_error"],
        "mean_species_auroc": species["auroc"],
        "mean_species_auprc": species["auprc"],
        "mean_species_calibration_error": species["calibration_error"],
        "residual_l2": regularization.get("residual_l2", 0.0),
        "availability_weight_l2": regularization.get(
            "availability_weight_l2", 0.0
        ),
        "species_bias_rms": parameters.get("species_bias_rms", np.nan),
        "effort_weight_rms": parameters.get("effort_weight_rms", np.nan),
        "availability_weight_mean": parameters.get(
            "availability_weight_mean", np.nan
        ),
        "availability_weight_deviation_rms": parameters.get(
            "availability_weight_deviation_rms", np.nan
        ),
    }


def species_frame(model_dir: Path, run_name: str) -> pd.DataFrame:
    frame = read_required_csv(model_dir / f"{run_name}_species_metrics.csv")
    frame = frame.loc[
        frame["model"].eq("two_component"),
        [
            "species_key",
            "common_name",
            "scientific_name",
            "auroc",
            "auprc",
            "calibration_error",
        ],
    ].copy()
    frame.insert(0, "run", run_name)
    return frame


def phenology_frame(model_dir: Path, run_name: str) -> pd.DataFrame:
    frame = read_required_csv(
        model_dir / f"{run_name}_focus_species_month.csv"
    )
    rows = []
    for common_name, group in frame.groupby("common_name", sort=True):
        rows.append(
            {
                "run": run_name,
                "common_name": common_name,
                "checklists": int(group["checklists"].sum()),
                "weighted_monthly_mae": weighted_mean(
                    group["two_component_calibration_error"],
                    group["checklists"],
                ),
                "months_better_than_availability": int(
                    (
                        group["two_component_calibration_error"]
                        < group["availability_only_calibration_error"]
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def response_frame(model_dir: Path, run_name: str) -> pd.DataFrame:
    frame = read_required_csv(
        model_dir / f"{run_name}_focus_species_response.csv"
    )
    rows = []
    for (common_name, covariate), group in frame.groupby(
        ["common_name", "covariate"], sort=True
    ):
        rows.append(
            {
                "run": run_name,
                "common_name": common_name,
                "covariate": covariate,
                "checklists": int(group["checklists"].sum()),
                "weighted_response_mae": weighted_mean(
                    group["two_component_calibration_error"],
                    group["checklists"],
                ),
                "shape_spearman": shape_spearman(
                    group["observed_detection_rate"],
                    group["two_component_predicted_detection_rate"],
                ),
                "bins_better_than_availability": int(
                    (
                        group["two_component_calibration_error"]
                        < group["availability_only_calibration_error"]
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def add_reference_deltas(
    frame: pd.DataFrame,
    reference_run: str,
    keys: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    reference = frame.loc[frame["run"].eq(reference_run), keys + metrics].copy()
    reference = reference.rename(
        columns={metric: f"{metric}_reference" for metric in metrics}
    )
    output = frame.merge(reference, on=keys, how="left", validate="many_to_one")
    for metric in metrics:
        output[f"delta_{metric}"] = (
            output[metric] - output[f"{metric}_reference"]
        )
    return output


def main() -> None:
    args = parse_args()
    if len(args.run) < 2:
        raise ValueError("Provide at least two --run arguments.")

    model_dir = Path(args.model_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else model_dir / "diagnostics" / "run_comparisons"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_run = args.run[0]

    runs = pd.DataFrame([run_summary(model_dir, run) for run in args.run])
    metric_columns = [
        "bce",
        "micro_auroc",
        "micro_auprc",
        "ece",
        "max_bin_error",
        "calibration_error",
        "mean_species_auroc",
        "mean_species_auprc",
        "mean_species_calibration_error",
    ]
    reference_metrics = runs.loc[
        runs["run"].eq(reference_run), metric_columns
    ].iloc[0]
    for metric in metric_columns:
        runs[f"delta_{metric}"] = runs[metric] - reference_metrics[metric]

    species = pd.concat(
        [species_frame(model_dir, run) for run in args.run],
        ignore_index=True,
    )
    species = add_reference_deltas(
        species,
        reference_run,
        ["species_key", "common_name", "scientific_name"],
        ["auroc", "auprc", "calibration_error"],
    )

    phenology = pd.concat(
        [phenology_frame(model_dir, run) for run in args.run],
        ignore_index=True,
    )
    phenology = add_reference_deltas(
        phenology,
        reference_run,
        ["common_name"],
        ["weighted_monthly_mae"],
    )

    response = pd.concat(
        [response_frame(model_dir, run) for run in args.run],
        ignore_index=True,
    )
    response = add_reference_deltas(
        response,
        reference_run,
        ["common_name", "covariate"],
        ["weighted_response_mae", "shape_spearman"],
    )

    stem = safe_filename_stem(args.run, args.comparison_name)
    runs.to_csv(output_dir / f"{stem}_runs.csv", index=False)
    species.to_csv(output_dir / f"{stem}_species.csv", index=False)
    phenology.to_csv(output_dir / f"{stem}_phenology.csv", index=False)
    response.to_csv(output_dir / f"{stem}_response.csv", index=False)

    print(f"Reference run: {reference_run}")
    print()
    print("Run comparison:")
    print(
        runs[
            [
                "run",
                "micro_auprc",
                "delta_micro_auprc",
                "ece",
                "delta_ece",
                "mean_species_auprc",
                "delta_mean_species_auprc",
                "mean_species_calibration_error",
                "delta_mean_species_calibration_error",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.5f}")
    )

    comparison_runs = [run for run in args.run if run != reference_run]
    for run_name in comparison_runs:
        print()
        print(f"Largest species AUPRC changes for {run_name}:")
        run_species = species.loc[species["run"].eq(run_name)]
        display_columns = [
            "common_name",
            "auprc_reference",
            "auprc",
            "delta_auprc",
            "delta_calibration_error",
        ]
        print("Gains:")
        print(
            run_species.nlargest(args.top, "delta_auprc")[
                display_columns
            ].to_string(index=False, float_format=lambda value: f"{value:.5f}")
        )
        print("Losses:")
        print(
            run_species.nsmallest(args.top, "delta_auprc")[
                display_columns
            ].to_string(index=False, float_format=lambda value: f"{value:.5f}")
        )

    print()
    print(f"Wrote comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
