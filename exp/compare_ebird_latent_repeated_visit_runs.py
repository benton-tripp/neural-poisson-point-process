"""
Compare saved latent repeated-visit model runs without retraining.

Run from the project root:

    python exp/compare_ebird_latent_repeated_visit_runs.py --runs latent_repeated_visit_e200 latent_repeated_visit_e200_mrate25 latent_repeated_visit_e200_mrate50 latent_repeated_visit_e200_mrate100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LATENT_DIR = "data/ebird/locality_season_top100/latent_models"
DEFAULT_BRIDGE_DIR = "data/ebird/locality_season_top100/detection_models"
DEFAULT_BRIDGE_RUN_NAME = "two_component_checklist_detection_e10_d20"
DEFAULT_RUNS = [
    "latent_repeated_visit_e200",
    "latent_repeated_visit_e200_mrate25",
    "latent_repeated_visit_e200_mrate50",
    "latent_repeated_visit_e200_mrate100",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare saved latent repeated-visit model runs."
    )
    parser.add_argument(
        "--latent-dir",
        default=DEFAULT_LATENT_DIR,
        help=f"Latent model output directory. Defaults to {DEFAULT_LATENT_DIR}.",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        default=DEFAULT_RUNS,
        help="Latent run names to compare, in plot order.",
    )
    parser.add_argument(
        "--bridge-dir",
        default=DEFAULT_BRIDGE_DIR,
        help=f"Two-component bridge output directory. Defaults to {DEFAULT_BRIDGE_DIR}.",
    )
    parser.add_argument(
        "--bridge-run-name",
        default=DEFAULT_BRIDGE_RUN_NAME,
        help=f"Bridge run name prefix. Defaults to {DEFAULT_BRIDGE_RUN_NAME}.",
    )
    parser.add_argument(
        "--comparison-name",
        default="latent_repeated_visit_mrate_sweep",
        help="Output subdirectory name when --output-dir is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to latent-dir/diagnostics/run_comparisons/comparison-name.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of species rows to print. Defaults to 15.",
    )
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required output: {path}")
    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_bridge(bridge_dir: Path, run_name: str) -> dict[str, pd.DataFrame]:
    species = read_required_csv(bridge_dir / f"{run_name}_species_metrics.csv")
    focus = read_optional_csv(bridge_dir / f"{run_name}_focus_species_season.csv")
    if "model" in species.columns:
        species = species.loc[species["model"] == "two_component"].copy()
    return {"species": species, "focus": focus}


def load_latent_run(latent_dir: Path, run_name: str) -> dict[str, pd.DataFrame]:
    return {
        "metrics": read_required_csv(latent_dir / f"{run_name}_metrics.csv"),
        "latent_diagnostics": read_optional_csv(
            latent_dir / f"{run_name}_latent_detection_diagnostics.csv"
        ),
        "availability": read_required_csv(
            latent_dir / f"{run_name}_availability_metrics.csv"
        ),
        "species": read_required_csv(latent_dir / f"{run_name}_species_metrics.csv"),
        "availability_species": read_required_csv(
            latent_dir / f"{run_name}_availability_species_metrics.csv"
        ),
        "focus_detection": read_required_csv(
            latent_dir / f"{run_name}_focus_species_season.csv"
        ),
        "focus_availability": read_required_csv(
            latent_dir / f"{run_name}_focus_species_availability_season.csv"
        ),
    }


def first_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return df.iloc[0]


def diagnostic_row(latent: dict[str, pd.DataFrame], model_name: str) -> pd.Series:
    diagnostics = latent["latent_diagnostics"]
    if diagnostics.empty:
        return pd.Series(dtype=float)
    rows = diagnostics.loc[diagnostics["model"] == model_name]
    return first_row(rows)


def build_run_summary(
    latent_dir: Path,
    runs: list[str],
    bridge: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    species_rows: list[pd.DataFrame] = []
    focus_rows: list[pd.DataFrame] = []
    focus_availability_rows: list[pd.DataFrame] = []
    availability_species_rows: list[pd.DataFrame] = []

    bridge_species = bridge["species"].copy()
    bridge_species = bridge_species.rename(
        columns={
            "auroc": "bridge_auroc",
            "auprc": "bridge_auprc",
            "calibration_error": "bridge_calibration_error",
            "mean_predicted_detection_rate": "bridge_mean_predicted_detection_rate",
        }
    )
    bridge_species_cols = [
        col
        for col in [
            "species_key",
            "common_name",
            "scientific_name",
            "bridge_auroc",
            "bridge_auprc",
            "bridge_calibration_error",
            "bridge_mean_predicted_detection_rate",
        ]
        if col in bridge_species.columns
    ]

    bridge_focus = bridge["focus"].copy()
    bridge_focus_cols = [
        col
        for col in [
            "common_name",
            "season_name",
            "two_component_predicted_detection_rate",
            "two_component_calibration_error",
            "availability_only_predicted_detection_rate",
            "availability_only_calibration_error",
        ]
        if col in bridge_focus.columns
    ]

    for run in runs:
        latent = load_latent_run(latent_dir, run)
        prior = first_row(latent["metrics"])
        posterior = diagnostic_row(
            latent, "latent_posterior_marginal_all_pairs_label_informed"
        )
        known_available = diagnostic_row(
            latent, "latent_conditional_detection_known_available_pairs"
        )
        availability = first_row(latent["availability"])

        summary = {
            "run_name": run,
            "prior_mean_predicted_detection_rate": prior.get(
                "mean_predicted_detection_rate", np.nan
            ),
            "observed_detection_rate": prior.get("observed_detection_rate", np.nan),
            "prior_calibration_error": prior.get("calibration_error", np.nan),
            "prior_bce": prior.get("bce", np.nan),
            "prior_micro_auroc": prior.get("micro_auroc", np.nan),
            "prior_micro_auprc": prior.get("micro_auprc", np.nan),
            "prior_ece": prior.get("ece", np.nan),
            "prior_max_bin_error": prior.get("max_bin_error", np.nan),
            "posterior_mean_predicted_detection_rate": posterior.get(
                "mean_predicted_detection_rate", np.nan
            ),
            "posterior_bce": posterior.get("bce", np.nan),
            "posterior_micro_auroc": posterior.get("micro_auroc", np.nan),
            "posterior_micro_auprc": posterior.get("micro_auprc", np.nan),
            "posterior_ece": posterior.get("ece", np.nan),
            "known_available_mean_predicted_detection_rate": known_available.get(
                "mean_predicted_detection_rate", np.nan
            ),
            "known_available_observed_detection_rate": known_available.get(
                "observed_detection_rate", np.nan
            ),
            "known_available_bce": known_available.get("bce", np.nan),
            "known_available_micro_auroc": known_available.get("micro_auroc", np.nan),
            "known_available_micro_auprc": known_available.get("micro_auprc", np.nan),
            "known_available_ece": known_available.get("ece", np.nan),
            "availability_mean_predicted": availability.get(
                "mean_predicted_availability", np.nan
            ),
            "availability_observed_positive_rate": availability.get(
                "observed_positive_rate", np.nan
            ),
            "availability_calibration_error_vs_observed_positive": availability.get(
                "calibration_error_vs_observed_positive", np.nan
            ),
            "availability_positive_triplet_auroc": availability.get(
                "positive_triplet_auroc", np.nan
            ),
            "availability_positive_triplet_auprc": availability.get(
                "positive_triplet_auprc", np.nan
            ),
        }
        summary["prior_underprediction"] = (
            summary["observed_detection_rate"]
            - summary["prior_mean_predicted_detection_rate"]
        )
        summaries.append(summary)

        species = latent["species"].copy()
        species["run_name"] = run
        species = species.merge(
            bridge_species[bridge_species_cols],
            on=[col for col in ["species_key", "common_name", "scientific_name"] if col in bridge_species_cols],
            how="left",
        )
        if "bridge_auprc" in species.columns:
            species["delta_auprc_vs_bridge"] = species["auprc"] - species["bridge_auprc"]
        if "bridge_auroc" in species.columns:
            species["delta_auroc_vs_bridge"] = species["auroc"] - species["bridge_auroc"]
        if "bridge_calibration_error" in species.columns:
            species["delta_calibration_error_vs_bridge"] = (
                species["calibration_error"] - species["bridge_calibration_error"]
            )
        species["latent_detection_underprediction"] = (
            species["observed_detection_rate"] - species["mean_predicted_detection_rate"]
        )
        species_rows.append(species)

        focus = latent["focus_detection"].copy()
        focus["run_name"] = run
        if not bridge_focus.empty:
            focus = focus.merge(
                bridge_focus[bridge_focus_cols],
                on=["common_name", "season_name"],
                how="left",
            )
            if "two_component_calibration_error" in focus.columns:
                focus["delta_abs_error_vs_two_component"] = (
                    focus["latent_marginal_calibration_error"]
                    - focus["two_component_calibration_error"]
                )
        focus_rows.append(focus)

        focus_availability = latent["focus_availability"].copy()
        focus_availability["run_name"] = run
        focus_availability_rows.append(focus_availability)

        availability_species = latent["availability_species"].copy()
        availability_species["run_name"] = run
        availability_species["availability_minus_observed_positive"] = (
            availability_species["mean_predicted_availability"]
            - availability_species["observed_positive_rate"]
        )
        availability_species_rows.append(availability_species)

    return (
        pd.DataFrame(summaries),
        pd.concat(species_rows, ignore_index=True),
        pd.concat(focus_rows, ignore_index=True),
        pd.concat(focus_availability_rows, ignore_index=True),
        pd.concat(availability_species_rows, ignore_index=True),
    )


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def add_focus_summary(run_summary: pd.DataFrame, focus: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, group in focus.groupby("run_name", sort=False):
        row = {
            "run_name": run,
            "focus_season_weighted_abs_error": weighted_mean(
                group["latent_marginal_calibration_error"], group["checklists"]
            ),
        }
        if "two_component_calibration_error" in group.columns:
            row["bridge_focus_season_weighted_abs_error"] = weighted_mean(
                group["two_component_calibration_error"], group["checklists"]
            )
            row["focus_season_delta_abs_error_vs_bridge"] = (
                row["focus_season_weighted_abs_error"]
                - row["bridge_focus_season_weighted_abs_error"]
            )
        rows.append(row)
    return run_summary.merge(pd.DataFrame(rows), on="run_name", how="left")


def write_plots(output_dir: Path, summary: pd.DataFrame, species: pd.DataFrame, focus: pd.DataFrame) -> None:
    x = np.arange(len(summary))
    labels = summary["run_name"].str.replace("latent_repeated_visit_", "", regex=False)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    plot_specs = [
        ("prior_micro_auprc", "Prior marginal AUPRC"),
        ("prior_bce", "Prior marginal BCE"),
        ("prior_ece", "Prior marginal ECE"),
        ("prior_mean_predicted_detection_rate", "Mean predicted detection rate"),
    ]
    for ax, (column, title) in zip(axes.flat, plot_specs):
        ax.plot(x, summary[column], marker="o")
        if column == "prior_mean_predicted_detection_rate":
            observed = summary["observed_detection_rate"].iloc[0]
            ax.axhline(observed, color="0.4", linestyle="--", linewidth=1, label="observed")
            ax.legend(frameon=False)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(True, alpha=0.25)
    fig.savefig(output_dir / "latent_run_tradeoffs.png", dpi=180)
    plt.close(fig)

    if "delta_auprc_vs_bridge" in species.columns:
        pivot_values = [
            species.loc[species["run_name"] == run, "delta_auprc_vs_bridge"].dropna()
            for run in summary["run_name"]
        ]
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        ax.boxplot(pivot_values, showfliers=False)
        ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
        ax.set_title("Species AUPRC delta versus two-component bridge")
        ax.set_ylabel("AUPRC delta")
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        fig.savefig(output_dir / "species_auprc_delta_boxplot.png", dpi=180)
        plt.close(fig)

    focus_summary = (
        focus.groupby("run_name", sort=False)
        .apply(lambda g: weighted_mean(g["latent_marginal_calibration_error"], g["checklists"]))
        .reindex(summary["run_name"])
    )
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(x, focus_summary.values, marker="o")
    if "two_component_calibration_error" in focus.columns:
        bridge_error = weighted_mean(
            focus.drop_duplicates(["common_name", "season_name"])[
                "two_component_calibration_error"
            ],
            focus.drop_duplicates(["common_name", "season_name"])["checklists"],
        )
        ax.axhline(bridge_error, color="0.4", linestyle="--", linewidth=1, label="bridge")
        ax.legend(frameon=False)
    ax.set_title("Focus-species season weighted absolute error")
    ax.set_ylabel("Weighted absolute error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(True, alpha=0.25)
    fig.savefig(output_dir / "focus_species_season_error.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    latent_dir = Path(args.latent_dir)
    bridge_dir = Path(args.bridge_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else latent_dir / "diagnostics" / "run_comparisons" / args.comparison_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge = load_bridge(bridge_dir, args.bridge_run_name)
    summary, species, focus, focus_availability, availability_species = build_run_summary(
        latent_dir, args.runs, bridge
    )
    summary = add_focus_summary(summary, focus)

    summary.to_csv(output_dir / "latent_run_summary.csv", index=False)
    species.to_csv(output_dir / "latent_species_vs_bridge.csv", index=False)
    focus.to_csv(output_dir / "latent_focus_species_season.csv", index=False)
    focus_availability.to_csv(
        output_dir / "latent_focus_species_availability_season.csv", index=False
    )
    availability_species.to_csv(output_dir / "latent_availability_species.csv", index=False)

    if "delta_auprc_vs_bridge" in species.columns:
        pivot = species.pivot_table(
            index=["species_key", "common_name", "scientific_name"],
            columns="run_name",
            values="delta_auprc_vs_bridge",
        ).reset_index()
        pivot.to_csv(output_dir / "latent_species_auprc_delta_pivot.csv", index=False)

    write_plots(output_dir, summary, species, focus)

    metadata = {
        "runs": args.runs,
        "latent_dir": str(latent_dir),
        "bridge_dir": str(bridge_dir),
        "bridge_run_name": args.bridge_run_name,
        "outputs": sorted(path.name for path in output_dir.iterdir()),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote latent run comparison to {output_dir}")
    print("\nRun summary:")
    summary_cols = [
        "run_name",
        "prior_micro_auprc",
        "prior_bce",
        "prior_calibration_error",
        "prior_ece",
        "availability_positive_triplet_auprc",
        "focus_season_weighted_abs_error",
    ]
    print(summary[summary_cols].to_string(index=False, float_format="%.5f"))

    if "delta_auprc_vs_bridge" in species.columns:
        print("\nLargest species AUPRC gains versus bridge across compared runs:")
        print(
            species.nlargest(args.top, "delta_auprc_vs_bridge")[
                ["run_name", "common_name", "auprc", "bridge_auprc", "delta_auprc_vs_bridge"]
            ].to_string(index=False, float_format="%.5f")
        )
        print("\nLargest species AUPRC losses versus bridge across compared runs:")
        print(
            species.nsmallest(args.top, "delta_auprc_vs_bridge")[
                ["run_name", "common_name", "auprc", "bridge_auprc", "delta_auprc_vs_bridge"]
            ].to_string(index=False, float_format="%.5f")
        )


if __name__ == "__main__":
    main()
