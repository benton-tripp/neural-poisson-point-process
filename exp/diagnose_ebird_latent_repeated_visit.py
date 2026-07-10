"""
Summarize saved latent repeated-visit model outputs.

Run from the project root:

    python exp/diagnose_ebird_latent_repeated_visit.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_repeated_visit_e100 --bridge-dir data/ebird/locality_season_top100/detection_models --bridge-run-name two_component_checklist_detection_e10_d20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LATENT_DIR = "data/ebird/locality_season_top100/latent_models"
DEFAULT_RUN_NAME = "latent_repeated_visit_e100"
DEFAULT_BRIDGE_DIR = "data/ebird/locality_season_top100/detection_models"
DEFAULT_BRIDGE_RUN_NAME = "two_component_checklist_detection_e10_d20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose saved latent repeated-visit model outputs."
    )
    parser.add_argument(
        "--latent-dir",
        default=DEFAULT_LATENT_DIR,
        help=f"Latent model output directory. Defaults to {DEFAULT_LATENT_DIR}.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help=f"Latent run name prefix. Defaults to {DEFAULT_RUN_NAME}.",
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
        "--compare-run",
        default=None,
        help="Optional earlier latent run name to compare against.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to latent-dir/diagnostics/run-name.",
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
        raise FileNotFoundError(f"Missing required output: {path}")
    return pd.read_csv(path)


def read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_latent(latent_dir: Path, run_name: str) -> dict[str, pd.DataFrame]:
    return {
        "metrics": read_required_csv(latent_dir / f"{run_name}_metrics.csv"),
        "species": read_required_csv(latent_dir / f"{run_name}_species_metrics.csv"),
        "availability": read_required_csv(
            latent_dir / f"{run_name}_availability_metrics.csv"
        ),
        "availability_species": read_required_csv(
            latent_dir / f"{run_name}_availability_species_metrics.csv"
        ),
        "focus_detection": read_required_csv(
            latent_dir / f"{run_name}_focus_species_season.csv"
        ),
        "focus_availability": read_required_csv(
            latent_dir / f"{run_name}_focus_species_availability_season.csv"
        ),
        "latent_diagnostics": read_optional_csv(
            latent_dir / f"{run_name}_latent_detection_diagnostics.csv"
        ),
    }


def load_bridge(bridge_dir: Path, run_name: str) -> dict[str, pd.DataFrame]:
    return {
        "metrics": read_required_csv(bridge_dir / f"{run_name}_metrics.csv"),
        "species": read_required_csv(bridge_dir / f"{run_name}_species_metrics.csv"),
        "focus": read_required_csv(bridge_dir / f"{run_name}_focus_species_season.csv"),
    }


def headline_comparison(
    latent: dict[str, pd.DataFrame],
    bridge: dict[str, pd.DataFrame],
    compare: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    rows = []

    latent_row = latent["metrics"].iloc[0].copy()
    latent_row["source"] = "latent_prior_marginal"
    rows.append(latent_row)

    if not latent["latent_diagnostics"].empty:
        for _, row in latent["latent_diagnostics"].iterrows():
            diag_row = row.copy()
            diag_row["checklists"] = np.nan
            diag_row["source"] = row["model"]
            rows.append(diag_row)

    for model in ["two_component", "availability_only", "effort_only", "train_prevalence"]:
        subset = bridge["metrics"].loc[bridge["metrics"]["model"] == model]
        if subset.empty:
            continue
        row = subset.iloc[0].copy()
        row["source"] = f"bridge_{model}"
        rows.append(row)

    if compare is not None:
        row = compare["metrics"].iloc[0].copy()
        row["source"] = "compare_latent_prior_marginal"
        rows.append(row)

    result = pd.DataFrame(rows)
    keep = [
        "source",
        "model",
        "checklists",
        "pairs",
        "detections",
        "observed_detection_rate",
        "mean_predicted_detection_rate",
        "calibration_error",
        "bce",
        "micro_auroc",
        "micro_auprc",
        "ece",
        "max_bin_error",
    ]
    return result[[col for col in keep if col in result.columns]]


def species_comparison(
    latent: dict[str, pd.DataFrame],
    bridge: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    latent_species = latent["species"].copy()
    bridge_species = bridge["species"].loc[
        bridge["species"]["model"].eq("two_component")
    ].copy()
    bridge_species = bridge_species[
        [
            "species_key",
            "common_name",
            "scientific_name",
            "auroc",
            "auprc",
            "calibration_error",
            "mean_predicted_detection_rate",
            "observed_detection_rate",
        ]
    ].rename(
        columns={
            "auroc": "bridge_auroc",
            "auprc": "bridge_auprc",
            "calibration_error": "bridge_calibration_error",
            "mean_predicted_detection_rate": "bridge_mean_predicted_detection_rate",
            "observed_detection_rate": "bridge_observed_detection_rate",
        }
    )
    merged = latent_species.merge(
        bridge_species,
        on=["species_key", "common_name", "scientific_name"],
        how="left",
    )
    merged = merged.rename(
        columns={
            "auroc": "latent_auroc",
            "auprc": "latent_auprc",
            "calibration_error": "latent_calibration_error",
            "mean_predicted_detection_rate": "latent_mean_predicted_detection_rate",
            "observed_detection_rate": "latent_observed_detection_rate",
        }
    )
    merged["delta_auprc_vs_bridge"] = merged["latent_auprc"] - merged["bridge_auprc"]
    merged["delta_auroc_vs_bridge"] = merged["latent_auroc"] - merged["bridge_auroc"]
    merged["delta_calibration_error_vs_bridge"] = (
        merged["latent_calibration_error"] - merged["bridge_calibration_error"]
    )
    merged["latent_detection_underprediction"] = (
        merged["latent_observed_detection_rate"]
        - merged["latent_mean_predicted_detection_rate"]
    )
    return merged.sort_values("delta_auprc_vs_bridge", ascending=False)


def availability_species_summary(latent: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = latent["availability_species"].copy()
    df["availability_minus_observed_positive"] = (
        df["mean_predicted_availability"] - df["observed_positive_rate"]
    )
    return df.sort_values("positive_triplet_auprc", ascending=False)


def focus_season_comparison(
    latent: dict[str, pd.DataFrame],
    bridge: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    latent_focus = latent["focus_detection"].copy()
    bridge_focus = bridge["focus"].copy()
    bridge_keep = [
        "common_name",
        "season_name",
        "two_component_predicted_detection_rate",
        "two_component_calibration_error",
        "availability_only_predicted_detection_rate",
        "availability_only_calibration_error",
    ]
    bridge_focus = bridge_focus[[col for col in bridge_keep if col in bridge_focus.columns]]
    merged = latent_focus.merge(
        bridge_focus,
        on=["common_name", "season_name"],
        how="left",
    )
    if "two_component_calibration_error" in merged.columns:
        merged["delta_abs_error_vs_two_component"] = (
            merged["latent_marginal_calibration_error"]
            - merged["two_component_calibration_error"]
        )
    if "availability_only_calibration_error" in merged.columns:
        merged["delta_abs_error_vs_availability_only"] = (
            merged["latent_marginal_calibration_error"]
            - merged["availability_only_calibration_error"]
        )
    return merged.sort_values("latent_marginal_calibration_error", ascending=False)


def run_delta(
    current: dict[str, pd.DataFrame],
    compare: dict[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    if compare is None:
        return pd.DataFrame()
    current_metrics = current["metrics"].iloc[0].copy()
    compare_metrics = compare["metrics"].iloc[0].copy()
    rows = []
    for metric in [
        "mean_predicted_detection_rate",
        "calibration_error",
        "bce",
        "micro_auroc",
        "micro_auprc",
        "ece",
        "max_bin_error",
    ]:
        rows.append(
            {
                "metric": metric,
                "current": float(current_metrics[metric]),
                "compare": float(compare_metrics[metric]),
                "delta": float(current_metrics[metric] - compare_metrics[metric]),
            }
        )
    current_avail = current["availability"].iloc[0].copy()
    compare_avail = compare["availability"].iloc[0].copy()
    for metric in [
        "mean_predicted_availability",
        "calibration_error_vs_observed_positive",
        "positive_triplet_auroc",
        "positive_triplet_auprc",
        "ece_vs_observed_positive",
        "max_bin_error_vs_observed_positive",
    ]:
        rows.append(
            {
                "metric": f"availability_{metric}",
                "current": float(current_avail[metric]),
                "compare": float(compare_avail[metric]),
                "delta": float(current_avail[metric] - compare_avail[metric]),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(output_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        if not table.empty:
            table.to_csv(output_dir / f"{name}.csv", index=False)


def main() -> None:
    args = parse_args()
    latent_dir = Path(args.latent_dir)
    bridge_dir = Path(args.bridge_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else latent_dir / "diagnostics" / args.run_name
    )

    latent = load_latent(latent_dir, args.run_name)
    bridge = load_bridge(bridge_dir, args.bridge_run_name)
    compare = load_latent(latent_dir, args.compare_run) if args.compare_run else None

    tables = {
        "headline_comparison": headline_comparison(latent, bridge, compare),
        "species_vs_bridge": species_comparison(latent, bridge),
        "availability_species": availability_species_summary(latent),
        "focus_species_season_vs_bridge": focus_season_comparison(latent, bridge),
        "run_delta": run_delta(latent, compare),
    }
    write_outputs(output_dir, tables)

    print(f"Wrote latent diagnostics to {output_dir}")
    print("\nHeadline comparison:")
    print(tables["headline_comparison"].to_string(index=False, float_format="%.5f"))

    if not tables["run_delta"].empty:
        print("\nChange versus comparison latent run:")
        print(tables["run_delta"].to_string(index=False, float_format="%.5f"))

    species = tables["species_vs_bridge"]
    print("\nLargest latent species AUPRC gains versus bridge:")
    cols = [
        "common_name",
        "latent_auprc",
        "bridge_auprc",
        "delta_auprc_vs_bridge",
        "latent_detection_underprediction",
    ]
    print(species[cols].head(args.top).to_string(index=False, float_format="%.5f"))

    print("\nLargest latent species AUPRC losses versus bridge:")
    print(
        species[cols]
        .tail(args.top)
        .sort_values("delta_auprc_vs_bridge")
        .to_string(index=False, float_format="%.5f")
    )

    focus = tables["focus_species_season_vs_bridge"]
    focus_cols = [
        "common_name",
        "season_name",
        "observed_detection_rate",
        "latent_marginal_predicted_detection_rate",
        "latent_marginal_calibration_error",
    ]
    if "two_component_predicted_detection_rate" in focus.columns:
        focus_cols.extend(
            ["two_component_predicted_detection_rate", "delta_abs_error_vs_two_component"]
        )
    print("\nWorst focus-species season latent marginal errors:")
    print(focus[focus_cols].head(args.top).to_string(index=False, float_format="%.5f"))


if __name__ == "__main__":
    main()
