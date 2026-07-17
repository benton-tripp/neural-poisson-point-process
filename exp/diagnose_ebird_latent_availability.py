"""Diagnose held-out latent availability for focus species.

Run from the project root after a latent-model run that writes
``<run_name>_focus_species_group_predictions.csv``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LATENT_DIR = "data/ebird/locality_season_top100/latent_models"
SEASON_ORDER = [
    "winter",
    "spring_migration",
    "early_breeding",
    "late_breeding",
    "fall_migration",
]
ENVIRONMENTAL_COLUMNS = {
    "canopy_median": "Canopy cover",
    "elevation_median": "Elevation",
    "distance_to_waterbody_m_median": "Distance to waterbody",
    "distance_to_coastline_m_median": "Distance to coastline",
}
REQUIRED_COLUMNS = {
    "common_name",
    "season_name",
    "n_checklists",
    "n_dates",
    "observed_any_detection",
    "predicted_availability",
    "conditional_any_detection_probability",
    "prior_any_detection_probability",
    *ENVIRONMENTAL_COLUMNS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose phenology, environment, and support for latent availability."
    )
    parser.add_argument("--latent-dir", default=DEFAULT_LATENT_DIR)
    parser.add_argument(
        "--dataset-dir",
        default=None,
        help=(
            "Locality-season dataset directory. Defaults to the parent of "
            "--latent-dir and is used to audit prior-year locality support."
        ),
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--environmental-bins", type=int, default=5)
    parser.add_argument("--top-cases", type=int, default=200)
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def aggregate_probabilities(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return (
        frame.groupby(keys, observed=True, dropna=False)
        .agg(
            locality_seasons=("observed_any_detection", "size"),
            observed_any_detection_rate=("observed_any_detection", "mean"),
            mean_predicted_availability=("predicted_availability", "mean"),
            mean_conditional_any_detection_probability=(
                "conditional_any_detection_probability",
                "mean",
            ),
            mean_prior_any_detection_probability=(
                "prior_any_detection_probability",
                "mean",
            ),
        )
        .reset_index()
        .assign(
            prior_any_detection_signed_error=lambda x: (
                x["mean_prior_any_detection_probability"]
                - x["observed_any_detection_rate"]
            ),
            availability_minus_observed_positive=lambda x: (
                x["mean_predicted_availability"]
                - x["observed_any_detection_rate"]
            ),
        )
    )


def summarize_phenology(frame: pd.DataFrame) -> pd.DataFrame:
    summary = aggregate_probabilities(frame, ["common_name", "season_name"])
    summary["season_name"] = pd.Categorical(
        summary["season_name"], categories=SEASON_ORDER, ordered=True
    )
    return summary.sort_values(["common_name", "season_name"]).reset_index(drop=True)


def summarize_environment(
    frame: pd.DataFrame, bin_count: int
) -> pd.DataFrame:
    rows = []
    for common_name, species_frame in frame.groupby("common_name", observed=True):
        for column, label in ENVIRONMENTAL_COLUMNS.items():
            work = species_frame.loc[species_frame[column].notna()].copy()
            if work.empty:
                continue
            work["environment_bin"] = pd.qcut(
                work[column], q=bin_count, labels=False, duplicates="drop"
            )
            work = work.loc[work["environment_bin"].notna()].copy()
            if work.empty:
                continue
            summary = aggregate_probabilities(
                work, ["common_name", "environment_bin"]
            )
            covariate = (
                work.groupby("environment_bin", observed=True)[column]
                .agg(covariate_min="min", covariate_mean="mean", covariate_max="max")
                .reset_index()
            )
            summary = summary.merge(covariate, on="environment_bin", how="left")
            summary.insert(1, "covariate", column)
            summary.insert(2, "covariate_label", label)
            rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_environment_diagnostics(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (common_name, covariate), work in summary.groupby(
        ["common_name", "covariate"], observed=True
    ):
        weights = work["locality_seasons"].to_numpy(dtype=float)
        rows.append(
            {
                "common_name": common_name,
                "covariate": covariate,
                "covariate_label": work["covariate_label"].iloc[0],
                "bins": int(len(work)),
                "locality_seasons": int(weights.sum()),
                "weighted_observable_mae": float(
                    np.average(
                        work["prior_any_detection_signed_error"].abs(),
                        weights=weights,
                    )
                ),
                "observable_shape_spearman": float(
                    work["observed_any_detection_rate"].corr(
                        work["mean_prior_any_detection_probability"],
                        method="spearman",
                    )
                ),
                "availability_shape_spearman": float(
                    work["observed_any_detection_rate"].corr(
                        work["mean_predicted_availability"], method="spearman"
                    )
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["weighted_observable_mae", "common_name"], ascending=[False, True]
    )


def summarize_high_support(
    frame: pd.DataFrame, top_cases: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    unique_groups = frame.drop_duplicates("locality_season_id")
    checklist_cutoff = float(unique_groups["n_checklists"].quantile(0.75))
    date_cutoff = float(unique_groups["n_dates"].quantile(0.75))
    high = frame.loc[
        (frame["n_checklists"] >= checklist_cutoff)
        & (frame["n_dates"] >= date_cutoff)
    ].copy()
    high["is_nondetection"] = high["observed_any_detection"].eq(0)
    summary = (
        high.groupby("common_name", observed=True)
        .agg(
            high_support_groups=("observed_any_detection", "size"),
            high_support_nondetections=("is_nondetection", "sum"),
            observed_any_detection_rate=("observed_any_detection", "mean"),
            mean_predicted_availability=("predicted_availability", "mean"),
            mean_prior_any_detection_probability=(
                "prior_any_detection_probability",
                "mean",
            ),
        )
        .reset_index()
    )
    summary["prior_any_detection_signed_error"] = (
        summary["mean_prior_any_detection_probability"]
        - summary["observed_any_detection_rate"]
    )
    nondetections = high.loc[high["is_nondetection"]].copy()
    extremes = (
        nondetections.groupby("common_name", observed=True)
        .agg(
            high_support_nondetections=("is_nondetection", "size"),
            mean_prior_any_detection_probability=(
                "prior_any_detection_probability",
                "mean",
            ),
            median_prior_any_detection_probability=(
                "prior_any_detection_probability",
                "median",
            ),
            max_prior_any_detection_probability=(
                "prior_any_detection_probability",
                "max",
            ),
            prior_any_ge_0p5=(
                "prior_any_detection_probability",
                lambda values: int(values.ge(0.5).sum()),
            ),
            prior_any_ge_0p8=(
                "prior_any_detection_probability",
                lambda values: int(values.ge(0.8).sum()),
            ),
            prior_any_ge_0p9=(
                "prior_any_detection_probability",
                lambda values: int(values.ge(0.9).sum()),
            ),
        )
        .reset_index()
        .sort_values(
            ["prior_any_ge_0p9", "max_prior_any_detection_probability"],
            ascending=False,
        )
    )
    cases = nondetections.sort_values(
        "prior_any_detection_probability", ascending=False
    ).head(top_cases)
    thresholds = {
        "n_checklists_q75": checklist_cutoff,
        "n_dates_q75": date_cutoff,
    }
    return summary, cases, extremes, thresholds


def enrich_cases_with_history(
    cases: pd.DataFrame, dataset_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    locality_path = dataset_dir / "locality_seasons.parquet"
    triplet_path = dataset_dir / "locality_season_species.parquet"
    if not locality_path.exists() or not triplet_path.exists():
        raise FileNotFoundError(
            "Historical support audit requires locality_seasons.parquet and "
            f"locality_season_species.parquet under {dataset_dir}"
        )

    locality_columns = [
        "locality_season_id",
        "locality_id",
        "locality",
        "locality_type",
        "county",
    ]
    localities = pd.read_parquet(locality_path, columns=locality_columns)
    enriched = cases.merge(localities, on="locality_season_id", how="left")
    if enriched["locality_id"].isna().any():
        raise ValueError("Some high-support cases did not map to a locality.")

    history_columns = [
        "locality_season_id",
        "locality_id",
        "season_year",
        "season_name",
        "species_key",
        "n_checklists",
        "n_detections",
    ]
    history = pd.read_parquet(triplet_path, columns=history_columns)
    history = history.loc[
        history["locality_id"].isin(enriched["locality_id"].unique())
        & history["species_key"].isin(enriched["species_key"].unique())
    ].copy()

    enriched = enriched.reset_index(drop=True)
    enriched.insert(0, "case_id", np.arange(len(enriched), dtype=np.int64))
    case_keys = enriched[
        ["case_id", "locality_id", "species_key", "season_year", "season_name"]
    ]
    joined = case_keys.merge(
        history,
        on=["locality_id", "species_key"],
        how="left",
        suffixes=("_case", "_history"),
    )
    prior = joined.loc[
        joined["season_year_history"].lt(joined["season_year_case"])
    ].copy()

    def aggregate_prior(work: pd.DataFrame, prefix: str) -> pd.DataFrame:
        count_columns = [
            f"{prefix}_groups",
            f"{prefix}_checklists",
            f"{prefix}_detections",
            f"{prefix}_positive_groups",
            f"{prefix}_latest_year_checklists",
            f"{prefix}_latest_year_detections",
        ]
        if work.empty:
            output = pd.DataFrame({"case_id": enriched["case_id"]})
            for column in count_columns:
                output[column] = 0
            output[f"{prefix}_latest_group_year"] = np.nan
            output[f"{prefix}_latest_positive_year"] = np.nan
            return output
        output = (
            work.groupby("case_id", observed=True)
            .agg(
                **{
                    f"{prefix}_groups": ("locality_season_id", "nunique"),
                    f"{prefix}_checklists": ("n_checklists", "sum"),
                    f"{prefix}_detections": ("n_detections", "sum"),
                    f"{prefix}_positive_groups": (
                        "n_detections",
                        lambda values: int(values.gt(0).sum()),
                    ),
                }
            )
            .reset_index()
        )
        latest_year = (
            work.groupby("case_id", observed=True)["season_year_history"]
            .max()
            .rename(f"{prefix}_latest_group_year")
            .reset_index()
        )
        latest_rows = work.merge(
            latest_year,
            left_on=["case_id", "season_year_history"],
            right_on=["case_id", f"{prefix}_latest_group_year"],
            how="inner",
        )
        latest_support = (
            latest_rows.groupby("case_id", observed=True)
            .agg(
                **{
                    f"{prefix}_latest_year_checklists": (
                        "n_checklists",
                        "sum",
                    ),
                    f"{prefix}_latest_year_detections": (
                        "n_detections",
                        "sum",
                    ),
                }
            )
            .reset_index()
        )
        latest_positive = (
            work.loc[work["n_detections"].gt(0)]
            .groupby("case_id", observed=True)["season_year_history"]
            .max()
            .rename(f"{prefix}_latest_positive_year")
            .reset_index()
        )
        return (
            output.merge(latest_year, on="case_id", how="left")
            .merge(latest_support, on="case_id", how="left")
            .merge(latest_positive, on="case_id", how="left")
        )

    prior_all = aggregate_prior(prior, "prior_all_season")
    prior_same = aggregate_prior(
        prior.loc[prior["season_name_history"].eq(prior["season_name_case"])],
        "prior_same_season",
    )
    enriched = enriched.merge(prior_all, on="case_id", how="left").merge(
        prior_same, on="case_id", how="left"
    )
    history_count_columns = [
        column
        for column in enriched.columns
        if (
            column.startswith("prior_all_season_")
            or column.startswith("prior_same_season_")
        )
        and not column.endswith("_year")
    ]
    enriched[history_count_columns] = (
        enriched[history_count_columns].fillna(0).astype(int)
    )
    for prefix in ("prior_all_season", "prior_same_season"):
        latest_positive = f"{prefix}_latest_positive_year"
        enriched[f"{prefix}_years_since_positive"] = (
            enriched["season_year"] - enriched[latest_positive]
        )
    enriched["history_class"] = np.select(
        [
            enriched["prior_same_season_positive_groups"].gt(0),
            enriched["prior_same_season_groups"].gt(0),
            enriched["prior_all_season_positive_groups"].gt(0),
            enriched["prior_all_season_groups"].gt(0),
        ],
        [
            "prior_same_season_detected",
            "prior_same_season_never_detected",
            "prior_other_season_detected",
            "prior_locality_never_detected",
        ],
        default="no_prior_locality_support",
    )

    history_summary = (
        enriched.groupby(["history_class", "locality_type"], dropna=False)
        .agg(
            cases=("case_id", "size"),
            species=("species_key", "nunique"),
            localities=("locality_id", "nunique"),
            mean_predicted_availability=("predicted_availability", "mean"),
            mean_prior_any_detection_probability=(
                "prior_any_detection_probability",
                "mean",
            ),
            median_test_checklists=("n_checklists", "median"),
            median_test_dates=("n_dates", "median"),
            median_unique_observers=("unique_observers", "median"),
        )
        .reset_index()
        .sort_values("cases", ascending=False)
    )
    return enriched.drop(columns="case_id"), history_summary


def plot_phenology(summary: pd.DataFrame, output_path: Path) -> None:
    species_names = sorted(summary["common_name"].unique())
    columns = 2
    rows = int(np.ceil(len(species_names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.2 * rows), squeeze=False)
    x = np.arange(len(SEASON_ORDER))
    for axis, common_name in zip(axes.ravel(), species_names):
        work = summary.loc[summary["common_name"].eq(common_name)].set_index(
            "season_name"
        )
        work = work.reindex(SEASON_ORDER)
        axis.plot(x, work["mean_predicted_availability"], marker="o", label="Availability psi")
        axis.plot(x, work["observed_any_detection_rate"], marker="o", label="Observed any detection")
        axis.plot(x, work["mean_prior_any_detection_probability"], marker="o", label="Predicted any detection")
        axis.set_title(common_name)
        axis.set_xticks(x, [value.replace("_", " ") for value in SEASON_ORDER], rotation=30, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[len(species_names) :]:
        axis.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Held-out focus-species phenology", y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_environment(summary: pd.DataFrame, output_dir: Path) -> None:
    for common_name, species_frame in summary.groupby("common_name", observed=True):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), squeeze=False)
        for axis, (column, label) in zip(axes.ravel(), ENVIRONMENTAL_COLUMNS.items()):
            work = species_frame.loc[species_frame["covariate"].eq(column)].sort_values(
                "covariate_mean"
            )
            axis.plot(work["covariate_mean"], work["mean_predicted_availability"], marker="o", label="Availability psi")
            axis.plot(work["covariate_mean"], work["observed_any_detection_rate"], marker="o", label="Observed any detection")
            axis.plot(work["covariate_mean"], work["mean_prior_any_detection_probability"], marker="o", label="Predicted any detection")
            axis.set_title(label)
            axis.grid(alpha=0.2)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.suptitle(common_name, y=1.01)
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{slugify(common_name)}_availability_environment.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)


def main() -> None:
    args = parse_args()
    latent_dir = Path(args.latent_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else latent_dir.parent
    input_path = latent_dir / f"{args.run_name}_focus_species_group_predictions.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing focus-species group predictions: {input_path}")
    frame = pd.read_csv(input_path)
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Focus-species group predictions are missing columns: {missing}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else latent_dir / "diagnostics" / "availability" / args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    phenology = summarize_phenology(frame)
    environment = summarize_environment(frame, args.environmental_bins)
    environment_diagnostics = (
        summarize_environment_diagnostics(environment)
        if not environment.empty
        else pd.DataFrame()
    )
    (
        high_support,
        high_support_cases,
        high_support_extremes,
        thresholds,
    ) = summarize_high_support(frame, args.top_cases)
    high_support_cases, history_summary = enrich_cases_with_history(
        high_support_cases, dataset_dir
    )

    phenology.to_csv(output_dir / "phenology_summary.csv", index=False)
    environment.to_csv(output_dir / "environmental_response_summary.csv", index=False)
    environment_diagnostics.to_csv(
        output_dir / "environmental_response_diagnostics.csv", index=False
    )
    high_support.to_csv(output_dir / "high_support_summary.csv", index=False)
    high_support_extremes.to_csv(
        output_dir / "high_support_nondetection_extremes.csv", index=False
    )
    high_support_cases.to_csv(
        output_dir / "high_support_nondetection_cases.csv", index=False
    )
    history_summary.to_csv(
        output_dir / "high_support_nondetection_history_summary.csv", index=False
    )
    (output_dir / "diagnostic_metadata.json").write_text(
        json.dumps(
            {
                "run_name": args.run_name,
                "dataset_dir": str(dataset_dir),
                "focus_species": sorted(frame["common_name"].unique()),
                "locality_seasons": int(frame["locality_season_id"].nunique()),
                "environmental_bins": int(args.environmental_bins),
                "high_support_thresholds": thresholds,
                "high_support_history_cases": int(len(high_support_cases)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_phenology(phenology, output_dir / "focus_species_phenology.png")
    if not environment.empty:
        plot_environment(environment, output_dir)

    print(f"Wrote latent-availability diagnostics to {output_dir}")
    print("\nLargest phenology-bin observable any-detection errors:")
    print(
        phenology.reindex(
            phenology["prior_any_detection_signed_error"].abs().sort_values(
                ascending=False
            ).index
        )
        .head(20)
        .to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nHigh-support summary:")
    print(
        high_support.sort_values(
            "prior_any_detection_signed_error", key=lambda values: values.abs(), ascending=False
        ).to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print("\nHigh-confidence zero-detection cases by species:")
    print(
        high_support_extremes.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print("\nTop zero-detection cases by prior locality history:")
    print(
        history_summary.to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    if not environment_diagnostics.empty:
        environment_overall = (
            environment_diagnostics.groupby("covariate", observed=True)
            .agg(
                mean_species_weighted_observable_mae=(
                    "weighted_observable_mae",
                    "mean",
                ),
                mean_observable_shape_spearman=(
                    "observable_shape_spearman",
                    "mean",
                ),
                mean_availability_shape_spearman=(
                    "availability_shape_spearman",
                    "mean",
                ),
            )
            .reset_index()
        )
        print("\nEnvironmental-response diagnostics across focus species:")
        print(
            environment_overall.to_string(
                index=False, float_format=lambda value: f"{value:.4f}"
            )
        )


if __name__ == "__main__":
    main()
