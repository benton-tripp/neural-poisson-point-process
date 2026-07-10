"""
Summarize persistent species-level patterns across latent repeated-visit runs.

Run from the project root:

    python exp/diagnose_ebird_latent_species_patterns.py --comparison-dir data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_mrate_srate_sweep
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_COMPARISON_DIR = (
    "data/ebird/locality_season_top100/latent_models/diagnostics/"
    "run_comparisons/latent_e200_mrate_srate_sweep"
)

SPECIES_GROUP_RULES = [
    (
        "water_coastal",
        [
            "gull",
            "tern",
            "pelican",
            "cormorant",
            "duck",
            "mallard",
            "merganser",
            "grebe",
            "bufflehead",
            "heron",
            "egret",
            "ibis",
            "kingfisher",
            "sandpiper",
            "plover",
            "rail",
            "coot",
            "gallinule",
            "loon",
        ],
    ),
    (
        "forest_woodland",
        [
            "warbler",
            "vireo",
            "woodpecker",
            "nuthatch",
            "chickadee",
            "titmouse",
            "wren",
            "thrush",
            "towhee",
            "tanager",
            "gnatcatcher",
            "kinglet",
            "wood-pewee",
            "flycatcher",
            "creeper",
        ],
    ),
    (
        "open_agricultural",
        [
            "meadowlark",
            "swallow",
            "blackbird",
            "grackle",
            "starling",
            "sparrow",
            "bobwhite",
            "killdeer",
            "cowbird",
            "kingbird",
        ],
    ),
    (
        "urban_generalist",
        [
            "house",
            "rock pigeon",
            "mourning dove",
            "cardinal",
            "robin",
            "crow",
            "blue jay",
            "mockingbird",
        ],
    ),
    (
        "raptor_scavenger",
        ["hawk", "eagle", "vulture", "owl", "falcon", "kite", "osprey"],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose broad species/regime patterns from saved latent run "
            "comparison outputs."
        )
    )
    parser.add_argument(
        "--comparison-dir",
        default=DEFAULT_COMPARISON_DIR,
        help=f"Saved latent run-comparison directory. Defaults to {DEFAULT_COMPARISON_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to comparison-dir/species_patterns.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of ranked rows to print. Defaults to 15.",
    )
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required comparison output: {path}")
    return pd.read_csv(path)


def infer_species_group(common_name: object) -> str:
    name = str(common_name).lower()
    for group, keywords in SPECIES_GROUP_RULES:
        if any(keyword in name for keyword in keywords):
            return group
    return "other"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values.loc[mask], weights=weights.loc[mask]))


def build_species_summary(species: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    species = species.copy()
    species["inferred_species_group"] = species["common_name"].map(infer_species_group)
    species["is_loss"] = species["delta_auprc_vs_bridge"] < 0
    species["is_gain"] = species["delta_auprc_vs_bridge"] > 0

    summary = (
        species.groupby(["species_key", "common_name", "scientific_name"], dropna=False)
        .agg(
            inferred_species_group=("inferred_species_group", "first"),
            runs=("run_name", "nunique"),
            min_delta_auprc_vs_bridge=("delta_auprc_vs_bridge", "min"),
            mean_delta_auprc_vs_bridge=("delta_auprc_vs_bridge", "mean"),
            max_delta_auprc_vs_bridge=("delta_auprc_vs_bridge", "max"),
            loss_run_count=("is_loss", "sum"),
            gain_run_count=("is_gain", "sum"),
            mean_latent_underprediction=("latent_detection_underprediction", "mean"),
            max_latent_underprediction=("latent_detection_underprediction", "max"),
            mean_observed_detection_rate=("observed_detection_rate", "mean"),
            mean_bridge_auprc=("bridge_auprc", "mean"),
            mean_latent_auprc=("auprc", "mean"),
        )
        .reset_index()
    )

    if not availability.empty:
        availability = availability.copy()
        availability["availability_minus_observed_positive"] = (
            availability["mean_predicted_availability"]
            - availability["observed_positive_rate"]
        )
        availability_summary = (
            availability.groupby(["species_key", "common_name", "scientific_name"], dropna=False)
            .agg(
                mean_availability_minus_observed_positive=(
                    "availability_minus_observed_positive",
                    "mean",
                ),
                max_availability_minus_observed_positive=(
                    "availability_minus_observed_positive",
                    "max",
                ),
                mean_observed_positive_rate=("observed_positive_rate", "mean"),
                mean_predicted_availability=("mean_predicted_availability", "mean"),
                mean_availability_auprc=("positive_triplet_auprc", "mean"),
            )
            .reset_index()
        )
        summary = summary.merge(
            availability_summary,
            on=["species_key", "common_name", "scientific_name"],
            how="left",
        )

    summary["loss_run_fraction"] = summary["loss_run_count"] / summary["runs"]
    summary["gain_run_fraction"] = summary["gain_run_count"] / summary["runs"]
    return summary.sort_values("mean_delta_auprc_vs_bridge")


def build_group_summary(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby("inferred_species_group", dropna=False)
        .agg(
            species=("species_key", "count"),
            mean_delta_auprc_vs_bridge=("mean_delta_auprc_vs_bridge", "mean"),
            median_delta_auprc_vs_bridge=("mean_delta_auprc_vs_bridge", "median"),
            min_delta_auprc_vs_bridge=("min_delta_auprc_vs_bridge", "min"),
            max_delta_auprc_vs_bridge=("max_delta_auprc_vs_bridge", "max"),
            mean_loss_run_fraction=("loss_run_fraction", "mean"),
            persistent_loss_species=("loss_run_fraction", lambda s: int((s == 1).sum())),
            mean_latent_underprediction=("mean_latent_underprediction", "mean"),
            mean_availability_minus_observed_positive=(
                "mean_availability_minus_observed_positive",
                "mean",
            ),
            mean_observed_detection_rate=("mean_observed_detection_rate", "mean"),
        )
        .reset_index()
        .sort_values("mean_delta_auprc_vs_bridge")
    )


def build_focus_season_summary(focus: pd.DataFrame) -> pd.DataFrame:
    if focus.empty:
        return pd.DataFrame()
    focus = focus.copy()
    if "delta_abs_error_vs_two_component" not in focus.columns:
        focus["delta_abs_error_vs_two_component"] = np.nan
    return (
        focus.groupby(["common_name", "season_name"], dropna=False)
        .apply(
            lambda group: pd.Series(
                {
                    "runs": group["run_name"].nunique(),
                    "mean_checklists": group["checklists"].mean(),
                    "mean_observed_detection_rate": group[
                        "observed_detection_rate"
                    ].mean(),
                    "mean_latent_predicted_detection_rate": group[
                        "latent_marginal_predicted_detection_rate"
                    ].mean(),
                    "mean_latent_calibration_error": group[
                        "latent_marginal_calibration_error"
                    ].mean(),
                    "max_latent_calibration_error": group[
                        "latent_marginal_calibration_error"
                    ].max(),
                    "mean_delta_abs_error_vs_two_component": group[
                        "delta_abs_error_vs_two_component"
                    ].mean(),
                    "weighted_mean_latent_calibration_error": weighted_mean(
                        group["latent_marginal_calibration_error"],
                        group["checklists"],
                    ),
                }
            )
        )
        .reset_index()
        .sort_values("mean_latent_calibration_error", ascending=False)
    )


def write_plots(output_dir: Path, species: pd.DataFrame) -> None:
    group_order = (
        species.groupby("inferred_species_group")["mean_delta_auprc_vs_bridge"]
        .median()
        .sort_values()
        .index.tolist()
    )
    box_data = [
        species.loc[
            species["inferred_species_group"] == group, "mean_delta_auprc_vs_bridge"
        ].dropna()
        for group in group_order
    ]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.boxplot(box_data, tick_labels=group_order, showfliers=False)
    ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
    ax.set_title("Mean latent AUPRC delta versus bridge by species group")
    ax.set_ylabel("Mean AUPRC delta")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(output_dir / "species_group_delta_auprc.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    groups = group_order
    cmap = plt.get_cmap("tab10")
    for idx, group in enumerate(groups):
        rows = species.loc[species["inferred_species_group"] == group]
        ax.scatter(
            rows["mean_observed_detection_rate"],
            rows["mean_delta_auprc_vs_bridge"],
            s=40,
            alpha=0.75,
            label=group,
            color=cmap(idx % 10),
        )
    ax.axhline(0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel("Mean observed detection rate")
    ax.set_ylabel("Mean AUPRC delta versus bridge")
    ax.set_title("Persistent latent species losses versus detection prevalence")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output_dir / "persistent_loss_scatter.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    comparison_dir = Path(args.comparison_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else comparison_dir / "species_patterns"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    species = read_required_csv(comparison_dir / "latent_species_vs_bridge.csv")
    availability = read_required_csv(comparison_dir / "latent_availability_species.csv")
    focus = read_required_csv(comparison_dir / "latent_focus_species_season.csv")

    species_summary = build_species_summary(species, availability)
    group_summary = build_group_summary(species_summary)
    focus_summary = build_focus_season_summary(focus)

    worst_losses = species_summary.nsmallest(args.top, "mean_delta_auprc_vs_bridge")
    best_gains = species_summary.nlargest(args.top, "mean_delta_auprc_vs_bridge")

    species_summary.to_csv(output_dir / "persistent_species_summary.csv", index=False)
    group_summary.to_csv(output_dir / "species_group_summary.csv", index=False)
    worst_losses.to_csv(output_dir / "worst_persistent_losses.csv", index=False)
    best_gains.to_csv(output_dir / "best_persistent_gains.csv", index=False)
    if not focus_summary.empty:
        focus_summary.to_csv(
            output_dir / "focus_species_season_persistent_errors.csv", index=False
        )
    write_plots(output_dir, species_summary)

    metadata = {
        "comparison_dir": str(comparison_dir),
        "outputs": sorted(path.name for path in output_dir.iterdir()),
        "species_group_rules": {
            group: keywords for group, keywords in SPECIES_GROUP_RULES
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"Wrote latent species-pattern diagnostics to {output_dir}")
    print("\nSpecies group summary:")
    display_group_cols = [
        "inferred_species_group",
        "species",
        "mean_delta_auprc_vs_bridge",
        "median_delta_auprc_vs_bridge",
        "persistent_loss_species",
        "mean_loss_run_fraction",
        "mean_latent_underprediction",
        "mean_availability_minus_observed_positive",
    ]
    print(
        group_summary[display_group_cols].to_string(
            index=False, float_format="%.5f"
        )
    )

    print("\nWorst persistent latent species losses versus bridge:")
    loss_cols = [
        "common_name",
        "inferred_species_group",
        "mean_delta_auprc_vs_bridge",
        "min_delta_auprc_vs_bridge",
        "loss_run_count",
        "mean_latent_underprediction",
        "mean_availability_minus_observed_positive",
    ]
    print(worst_losses[loss_cols].to_string(index=False, float_format="%.5f"))

    print("\nBest persistent latent species gains versus bridge:")
    gain_cols = [
        "common_name",
        "inferred_species_group",
        "mean_delta_auprc_vs_bridge",
        "max_delta_auprc_vs_bridge",
        "gain_run_count",
        "mean_latent_underprediction",
        "mean_availability_minus_observed_positive",
    ]
    print(best_gains[gain_cols].to_string(index=False, float_format="%.5f"))

    if not focus_summary.empty:
        print("\nWorst focus-species season latent marginal errors across runs:")
        focus_cols = [
            "common_name",
            "season_name",
            "mean_observed_detection_rate",
            "mean_latent_predicted_detection_rate",
            "mean_latent_calibration_error",
            "mean_delta_abs_error_vs_two_component",
        ]
        print(
            focus_summary.head(args.top)[focus_cols].to_string(
                index=False, float_format="%.5f"
            )
        )


if __name__ == "__main__":
    main()
