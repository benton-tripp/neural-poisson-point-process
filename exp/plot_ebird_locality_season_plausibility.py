"""
Plot phenology and environmental-response diagnostics for the two-component
locality-season/checklist detection model.

Run from the project root:

    python exp/plot_ebird_locality_season_plausibility.py --model-dir data/ebird/locality_season_top100/detection_models --run-name two_component_checklist_detection_e10_d20
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MODEL_DIR = "data/ebird/locality_season_top100/detection_models"
DEFAULT_RUN_NAME = "two_component_checklist_detection_e10_d20"

MODEL_STYLES = {
    "observed": {
        "label": "Observed",
        "color": "#202020",
        "linestyle": "-",
        "linewidth": 2.4,
        "marker": "o",
    },
    "availability_only": {
        "label": "Availability only",
        "color": "#2878b5",
        "linestyle": "--",
        "linewidth": 1.8,
        "marker": "s",
    },
    "effort_only": {
        "label": "Effort only",
        "color": "#777777",
        "linestyle": ":",
        "linewidth": 1.6,
        "marker": "^",
    },
    "two_component": {
        "label": "Two component",
        "color": "#c43c35",
        "linestyle": "-",
        "linewidth": 2.0,
        "marker": "D",
    },
}

COVARIATE_ORDER = [
    "canopy_median",
    "elevation_median",
    "distance_to_waterbody_m_median",
    "distance_to_coastline_m_median",
]

COVARIATE_LABELS = {
    "canopy_median": "Canopy cover",
    "elevation_median": "Elevation (m)",
    "distance_to_waterbody_m_median": "Distance to waterbody (km)",
    "distance_to_coastline_m_median": "Distance to coastline (km)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot focus-species monthly phenology and environmental-response "
            "diagnostics from saved two-component outputs."
        )
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Detection model output directory. Defaults to {DEFAULT_MODEL_DIR}.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help=f"Run name prefix. Defaults to {DEFAULT_RUN_NAME}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory. Defaults to "
            "model-dir/diagnostics/plausibility/run-name."
        ),
    )
    parser.add_argument(
        "--species",
        nargs="*",
        default=None,
        help="Optional common names to plot. Defaults to every focus species.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required output: {path}")
    return pd.read_csv(path)


def select_species(
    month: pd.DataFrame,
    response: pd.DataFrame,
    requested: list[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    available = sorted(set(month["common_name"]) & set(response["common_name"]))
    if not requested:
        selected = available
    else:
        missing = sorted(set(requested) - set(available))
        if missing:
            raise ValueError(
                "Requested species not found in both diagnostic files: "
                + ", ".join(missing)
            )
        selected = requested
    return (
        month.loc[month["common_name"].isin(selected)].copy(),
        response.loc[response["common_name"].isin(selected)].copy(),
        selected,
    )


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(dtype=float), weights=weights))


def spearman_shape(observed: pd.Series, predicted: pd.Series) -> float:
    observed_rank = observed.rank(method="average")
    predicted_rank = predicted.rank(method="average")
    return float(observed_rank.corr(predicted_rank))


def phenology_summary(month: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for species, group in month.groupby("common_name", sort=True):
        weights = group["checklists"].to_numpy(dtype=float)
        row = {
            "common_name": species,
            "months": len(group),
            "checklists": int(group["checklists"].sum()),
        }
        for model in ["availability_only", "effort_only", "two_component"]:
            error_col = f"{model}_calibration_error"
            row[f"{model}_weighted_mae"] = weighted_mean(
                group[error_col], weights
            )
        row["two_component_delta_vs_availability"] = (
            row["two_component_weighted_mae"]
            - row["availability_only_weighted_mae"]
        )
        row["two_component_months_better_than_availability"] = int(
            (
                group["two_component_calibration_error"]
                < group["availability_only_calibration_error"]
            ).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        "two_component_delta_vs_availability"
    )


def response_summary(response: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (species, covariate), group in response.groupby(
        ["common_name", "covariate"], sort=True
    ):
        group = group.sort_values("bin_id")
        weights = group["checklists"].to_numpy(dtype=float)
        row = {
            "common_name": species,
            "covariate": covariate,
            "covariate_label": group["covariate_label"].iloc[0],
            "bins": len(group),
            "checklists": int(group["checklists"].sum()),
        }
        for model in ["availability_only", "effort_only", "two_component"]:
            error_col = f"{model}_calibration_error"
            predicted_col = f"{model}_predicted_detection_rate"
            row[f"{model}_weighted_mae"] = weighted_mean(
                group[error_col], weights
            )
            row[f"{model}_shape_spearman"] = spearman_shape(
                group["observed_detection_rate"],
                group[predicted_col],
            )
        row["two_component_delta_vs_availability"] = (
            row["two_component_weighted_mae"]
            - row["availability_only_weighted_mae"]
        )
        row["two_component_bins_better_than_availability"] = int(
            (
                group["two_component_calibration_error"]
                < group["availability_only_calibration_error"]
            ).sum()
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["common_name", "covariate"]
    )


def plot_series(
    ax: plt.Axes,
    x: np.ndarray,
    group: pd.DataFrame,
) -> None:
    for model in ["observed", "availability_only", "effort_only", "two_component"]:
        style = MODEL_STYLES[model]
        column = (
            "observed_detection_rate"
            if model == "observed"
            else f"{model}_predicted_detection_rate"
        )
        ax.plot(
            x,
            group[column],
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=4,
        )


def plot_phenology_overview(
    month: pd.DataFrame,
    species: list[str],
    output_path: Path,
) -> None:
    ncols = 2
    nrows = math.ceil(len(species) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14, max(4, 3.3 * nrows)),
        squeeze=False,
        sharex=True,
    )
    for ax, common_name in zip(axes.ravel(), species):
        group = month.loc[month["common_name"] == common_name].sort_values("month")
        plot_series(ax, group["month"].to_numpy(), group)
        ax.set_title(common_name)
        ax.set_ylabel("Detection probability")
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels([calendar.month_abbr[i] for i in range(1, 13)])
        ax.grid(alpha=0.22)
    for ax in axes.ravel()[len(species) :]:
        ax.set_visible(False)
    visible_axes = [ax for ax in axes.ravel() if ax.get_visible()]
    handles, labels = visible_axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Focus-species monthly phenology", fontsize=16)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def display_x(group: pd.DataFrame, covariate: str) -> np.ndarray:
    values = group["covariate_mean"].to_numpy(dtype=float)
    if covariate in {
        "distance_to_waterbody_m_median",
        "distance_to_coastline_m_median",
    }:
        values = values / 1000.0
    return values


def plot_species_response(
    response: pd.DataFrame,
    common_name: str,
    output_path: Path,
) -> None:
    species_response = response.loc[response["common_name"] == common_name]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, covariate in zip(axes.ravel(), COVARIATE_ORDER):
        group = species_response.loc[
            species_response["covariate"] == covariate
        ].sort_values("bin_id")
        if group.empty:
            ax.set_visible(False)
            continue
        plot_series(ax, display_x(group, covariate), group)
        ax.set_title(COVARIATE_LABELS[covariate])
        ax.set_ylabel("Detection probability")
        if covariate in {
            "distance_to_waterbody_m_median",
            "distance_to_coastline_m_median",
        }:
            ax.set_xscale("symlog", linthresh=1.0)
        ax.grid(alpha=0.22)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle(common_name, fontsize=16)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_response_delta_heatmap(
    summary: pd.DataFrame,
    species: list[str],
    output_path: Path,
) -> None:
    pivot = summary.pivot(
        index="common_name",
        columns="covariate",
        values="two_component_delta_vs_availability",
    ).reindex(index=species, columns=COVARIATE_ORDER)
    values = pivot.to_numpy(dtype=float)
    limit = np.nanmax(np.abs(values))
    if not np.isfinite(limit) or limit == 0:
        limit = 0.01
    fig, ax = plt.subplots(figsize=(11, max(5, 0.55 * len(species) + 2)))
    image = ax.imshow(
        values,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    ax.set_xticks(range(len(COVARIATE_ORDER)))
    ax.set_xticklabels(
        [
            "Canopy",
            "Elevation",
            "Water distance",
            "Coast distance",
        ],
        rotation=20,
        ha="right",
    )
    ax.set_yticks(range(len(species)))
    ax.set_yticklabels(species)
    ax.set_title("Two-component response MAE minus availability-only MAE")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(
                    col,
                    row,
                    f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Negative is better for two-component")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else model_dir / "diagnostics" / "plausibility" / args.run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    month = read_required_csv(
        model_dir / f"{args.run_name}_focus_species_month.csv"
    )
    response = read_required_csv(
        model_dir / f"{args.run_name}_focus_species_response.csv"
    )
    month, response, species = select_species(month, response, args.species)

    phenology = phenology_summary(month)
    response_metrics = response_summary(response)
    phenology.to_csv(output_dir / "phenology_summary.csv", index=False)
    response_metrics.to_csv(
        output_dir / "environmental_response_summary.csv",
        index=False,
    )

    plot_phenology_overview(
        month,
        species,
        output_dir / "phenology_overview.png",
    )
    for common_name in species:
        plot_species_response(
            response,
            common_name,
            output_dir / f"{slugify(common_name)}_environmental_response.png",
        )
    plot_response_delta_heatmap(
        response_metrics,
        species,
        output_dir / "environmental_response_mae_delta.png",
    )

    metadata = {
        "model_dir": str(model_dir),
        "run_name": args.run_name,
        "species": species,
        "phenology_rows": len(month),
        "environmental_response_rows": len(response),
        "output_dir": str(output_dir),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote plausibility diagnostics to {output_dir}")
    print()
    print("Monthly phenology weighted MAE:")
    print(
        phenology[
            [
                "common_name",
                "availability_only_weighted_mae",
                "two_component_weighted_mae",
                "two_component_delta_vs_availability",
                "two_component_months_better_than_availability",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    print()
    print("Environmental-response weighted MAE by covariate:")
    print(
        response_metrics.groupby("covariate")[
            [
                "availability_only_weighted_mae",
                "two_component_weighted_mae",
                "availability_only_shape_spearman",
                "two_component_shape_spearman",
            ]
        ]
        .mean()
        .to_string(float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
