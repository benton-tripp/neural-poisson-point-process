"""Evaluate fair latent-model observables across transfer-relevant strata.

This diagnostic uses held-out focus-species group predictions from a completed
latent repeated-visit run. It does not retrain the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnose_ebird_latent_availability import enrich_cases_with_history
from ebird_joint_tabular_baseline import auc_roc, average_precision


DEFAULT_LATENT_DIR = "data/ebird/locality_season_top100/latent_models"
EPS = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fair group any-detection predictions across locality-history, "
            "observer-diversity, locality-type, and season strata."
        )
    )
    parser.add_argument("--latent-dir", default=DEFAULT_LATENT_DIR)
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Defaults to the parent of --latent-dir.",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def expected_calibration_error(
    labels: np.ndarray, probabilities: np.ndarray, bin_count: int
) -> float:
    probabilities = np.clip(probabilities, 0.0, 1.0)
    bin_indices = np.minimum(
        (probabilities * bin_count).astype(np.int64), bin_count - 1
    )
    total = len(labels)
    error = 0.0
    for bin_index in range(bin_count):
        mask = bin_indices == bin_index
        if mask.any():
            error += (
                float(mask.sum())
                / total
                * abs(float(probabilities[mask].mean() - labels[mask].mean()))
            )
    return error


def binary_metrics(
    frame: pd.DataFrame,
    calibration_bins: int,
    probability_column: str = "prior_any_detection_probability",
) -> dict[str, float | int]:
    labels = frame["observed_any_detection"].to_numpy(dtype=float)
    probabilities = frame[probability_column].to_numpy(dtype=float)
    clipped = np.clip(probabilities, EPS, 1.0 - EPS)
    observed_rate = float(labels.mean())
    auprc = (
        average_precision(labels, probabilities)
        if labels.sum() > 0
        else np.nan
    )
    auroc = (
        auc_roc(labels, probabilities)
        if np.unique(labels).size == 2
        else np.nan
    )
    return {
        "pairs": int(len(frame)),
        "locality_seasons": int(frame["locality_season_id"].nunique()),
        "species": int(frame["species_key"].nunique()),
        "positive_pairs": int(labels.sum()),
        "observed_positive_rate": observed_rate,
        "mean_predicted_any_detection_probability": float(probabilities.mean()),
        "signed_calibration_error": float(probabilities.mean() - observed_rate),
        "brier": float(np.mean(np.square(probabilities - labels))),
        "bce": float(
            -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
        ),
        "auroc": auroc,
        "auprc": auprc,
        "auprc_lift_over_prevalence": (
            float(auprc / observed_rate) if observed_rate > 0 else np.nan
        ),
        "ece": expected_calibration_error(labels, probabilities, calibration_bins),
    }


def add_transfer_strata(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "locality_seen_in_training" in output.columns:
        seen_values = output["locality_seen_in_training"]
        if pd.api.types.is_bool_dtype(seen_values):
            seen_in_training = seen_values.fillna(False)
        else:
            seen_in_training = (
                seen_values.astype("string").str.lower().isin(["true", "1", "yes"])
            )
    else:
        seen_in_training = output["prior_all_season_groups"].gt(0)
    output["locality_history"] = np.where(
        seen_in_training, "seen_locality", "unseen_locality"
    )
    output["same_season_history"] = np.select(
        [
            output["prior_same_season_groups"].eq(0),
            output["prior_same_season_latest_year_detections"].gt(0),
            output["prior_same_season_positive_groups"].gt(0),
        ],
        [
            "no_prior_same_season",
            "detected_latest_prior_year",
            "past_detection_recent_zero",
        ],
        default="never_detected_same_season",
    )
    output["observer_diversity"] = pd.cut(
        output["unique_observers"],
        bins=[-np.inf, 1, 5, np.inf],
        labels=["1", "2-5", "6+"],
    ).astype("string")
    output["locality_type_stratum"] = output["locality_type"].fillna("unknown")
    output["season_stratum"] = output["season_name"].astype("string")
    return output


def summarize_strata(
    frame: pd.DataFrame,
    calibration_bins: int,
    probability_column: str = "prior_any_detection_probability",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stratum_columns = {
        "locality_history": "locality_history",
        "same_season_history": "same_season_history",
        "observer_diversity": "observer_diversity",
        "locality_type": "locality_type_stratum",
        "season": "season_stratum",
    }
    summary_rows = []
    species_rows = []
    for stratum_type, column in stratum_columns.items():
        for stratum, work in frame.groupby(column, observed=True, dropna=False):
            metrics = binary_metrics(work, calibration_bins, probability_column)
            per_species = []
            for common_name, species_frame in work.groupby(
                "common_name", observed=True
            ):
                species_metrics = binary_metrics(
                    species_frame, calibration_bins, probability_column
                )
                species_metrics.update(
                    {
                        "stratum_type": stratum_type,
                        "stratum": str(stratum),
                        "common_name": common_name,
                    }
                )
                per_species.append(species_metrics)
                species_rows.append(species_metrics)
            species_table = pd.DataFrame(per_species)
            metrics.update(
                {
                    "stratum_type": stratum_type,
                    "stratum": str(stratum),
                    "macro_auroc": float(species_table["auroc"].mean()),
                    "macro_auprc": float(species_table["auprc"].mean()),
                    "macro_auprc_lift_over_prevalence": float(
                        species_table["auprc_lift_over_prevalence"].mean()
                    ),
                    "mean_abs_species_calibration_error": float(
                        species_table["signed_calibration_error"].abs().mean()
                    ),
                }
            )
            summary_rows.append(metrics)
    summary = pd.DataFrame(summary_rows).sort_values(["stratum_type", "stratum"])
    species = pd.DataFrame(species_rows).sort_values(
        ["stratum_type", "stratum", "common_name"]
    )
    return summary, species


def main() -> None:
    args = parse_args()
    latent_dir = Path(args.latent_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else latent_dir.parent
    prediction_path = (
        latent_dir / f"{args.run_name}_focus_species_group_predictions.csv"
    )
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Missing focus-species group predictions: {prediction_path}"
        )
    frame = pd.read_csv(prediction_path)
    enriched, _ = enrich_cases_with_history(frame, dataset_dir)
    enriched = add_transfer_strata(enriched)
    summary, species = summarize_strata(enriched, args.calibration_bins)
    history_comparison = pd.DataFrame()
    history_species_comparison = pd.DataFrame()
    if "portable_prior_any_detection_probability" in enriched.columns:
        portable_summary, portable_species = summarize_strata(
            enriched,
            args.calibration_bins,
            "portable_prior_any_detection_probability",
        )
        history_comparison = pd.concat(
            [
                portable_summary.assign(model_variant="portable_no_history"),
                summary.assign(model_variant="history_adapted"),
            ],
            ignore_index=True,
        )
        history_species_comparison = pd.concat(
            [
                portable_species.assign(model_variant="portable_no_history"),
                species.assign(model_variant="history_adapted"),
            ],
            ignore_index=True,
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else latent_dir / "diagnostics" / "transfer_strata" / args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "transfer_strata_summary.csv", index=False)
    species.to_csv(output_dir / "transfer_strata_species.csv", index=False)
    if not history_comparison.empty:
        history_comparison.to_csv(
            output_dir / "history_adaptation_strata_summary.csv", index=False
        )
        history_species_comparison.to_csv(
            output_dir / "history_adaptation_strata_species.csv", index=False
        )
    (output_dir / "diagnostic_metadata.json").write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "dataset_dir": str(dataset_dir),
                "focus_species": sorted(enriched["common_name"].unique()),
                "locality_seasons": int(enriched["locality_season_id"].nunique()),
                "pairs": int(len(enriched)),
                "calibration_bins": int(args.calibration_bins),
                "history_adaptation_comparison": bool(
                    not history_comparison.empty
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote latent transfer-strata diagnostics to {output_dir}")
    for stratum_type in (
        "locality_history",
        "same_season_history",
        "observer_diversity",
        "locality_type",
    ):
        print(f"\n{stratum_type.replace('_', ' ').title()}:")
        columns = [
            "stratum",
            "locality_seasons",
            "pairs",
            "observed_positive_rate",
            "mean_predicted_any_detection_probability",
            "signed_calibration_error",
            "bce",
            "auprc",
            "auprc_lift_over_prevalence",
            "macro_auprc",
            "mean_abs_species_calibration_error",
        ]
        print(
            summary.loc[summary["stratum_type"].eq(stratum_type), columns].to_string(
                index=False, float_format=lambda value: f"{value:.4f}"
            )
        )
    if not history_comparison.empty:
        print("\nPortable vs history-adapted same-season results:")
        columns = [
            "model_variant",
            "stratum",
            "pairs",
            "observed_positive_rate",
            "mean_predicted_any_detection_probability",
            "signed_calibration_error",
            "bce",
            "macro_auprc",
            "mean_abs_species_calibration_error",
        ]
        print(
            history_comparison.loc[
                history_comparison["stratum_type"].eq("same_season_history"),
                columns,
            ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
        )


if __name__ == "__main__":
    main()
