"""
Summarize diagnostics for the locality-season/checklist detection bridge.

This reads the saved outputs from `ebird_locality_season_detection_model.py`
and creates quick comparison tables without retraining the model.

Run from the project root:

    python exp/diagnose_ebird_locality_season_detection.py --model-dir data/ebird/locality_season_top100/detection_models --run-name two_component_checklist_detection_e10_d20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODEL_DIR = "data/ebird/locality_season_top100/detection_models"
DEFAULT_RUN_NAME = "two_component_checklist_detection_e10_d20"
DEFAULT_OUTPUT_DIR_NAME = "diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose saved locality-season checklist detection outputs."
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing detection model outputs. Defaults to {DEFAULT_MODEL_DIR}.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help=f"Run name prefix. Defaults to {DEFAULT_RUN_NAME}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Diagnostics output directory. Defaults to model-dir/diagnostics.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Number of top rows to print/write for ranked diagnostics. Defaults to 15.",
    )
    parser.add_argument(
        "--min-bin-pairs",
        type=int,
        default=100,
        help=(
            "Minimum pairs required for probability-bin calibration rows in "
            "ranked worst-bin diagnostics. Defaults to 100."
        ),
    )
    return parser.parse_args()


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required output: {path}")
    return pd.read_csv(path)


def load_outputs(model_dir: Path, run_name: str) -> dict[str, pd.DataFrame]:
    outputs = {
        "metrics": read_required_csv(model_dir / f"{run_name}_metrics.csv"),
        "species_metrics": read_required_csv(
            model_dir / f"{run_name}_species_metrics.csv"
        ),
        "focus": read_required_csv(model_dir / f"{run_name}_focus_species_season.csv"),
    }
    delta_path = model_dir / f"{run_name}_species_delta_vs_availability.csv"
    if delta_path.exists():
        outputs["species_delta"] = pd.read_csv(delta_path)
    else:
        species_metrics = outputs["species_metrics"]
        outputs["species_delta"] = species_metrics.pivot(
            index=["species_key", "common_name", "scientific_name"],
            columns="model",
            values="auprc",
        ).reset_index()
        outputs["species_delta"]["delta_two_component_vs_availability"] = (
            outputs["species_delta"]["two_component"]
            - outputs["species_delta"]["availability_only"]
        )
    calibration_path = model_dir / f"{run_name}_calibration.csv"
    if calibration_path.exists():
        outputs["calibration"] = pd.read_csv(calibration_path)
    else:
        outputs["calibration"] = pd.DataFrame()
    strata_metrics_path = model_dir / f"{run_name}_strata_metrics.csv"
    if strata_metrics_path.exists():
        outputs["strata_metrics"] = pd.read_csv(strata_metrics_path)
    else:
        outputs["strata_metrics"] = pd.DataFrame()
    strata_deltas_path = model_dir / f"{run_name}_strata_deltas.csv"
    if strata_deltas_path.exists():
        outputs["strata_deltas"] = pd.read_csv(strata_deltas_path)
    else:
        outputs["strata_deltas"] = pd.DataFrame()
    county_season_metrics_path = model_dir / f"{run_name}_county_season_metrics.csv"
    if county_season_metrics_path.exists():
        outputs["county_season_metrics"] = pd.read_csv(county_season_metrics_path)
    else:
        outputs["county_season_metrics"] = pd.DataFrame()
    county_season_deltas_path = model_dir / f"{run_name}_county_season_deltas.csv"
    if county_season_deltas_path.exists():
        outputs["county_season_deltas"] = pd.read_csv(county_season_deltas_path)
    else:
        outputs["county_season_deltas"] = pd.DataFrame()
    county_season_calibration_path = (
        model_dir / f"{run_name}_county_season_calibration.csv"
    )
    if county_season_calibration_path.exists():
        outputs["county_season_calibration"] = pd.read_csv(
            county_season_calibration_path
        )
    else:
        outputs["county_season_calibration"] = pd.DataFrame()
    focus_season_calibration_path = (
        model_dir / f"{run_name}_focus_species_season_calibration.csv"
    )
    if focus_season_calibration_path.exists():
        outputs["focus_season_calibration"] = pd.read_csv(
            focus_season_calibration_path
        )
    else:
        outputs["focus_season_calibration"] = pd.DataFrame()
    return outputs


def metric_delta(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "bce",
        "micro_auroc",
        "micro_auprc",
        "ece",
        "calibration_error",
        "max_bin_error",
    ]
    wide = metrics.set_index("model")
    rows = []
    if "two_component" not in wide.index:
        return pd.DataFrame()
    for baseline in ["availability_only", "effort_only", "train_prevalence"]:
        if baseline not in wide.index:
            continue
        row = {
            "candidate_model": "two_component",
            "baseline_model": baseline,
        }
        for col in numeric:
            if col in wide.columns:
                row[f"candidate_{col}"] = float(wide.loc["two_component", col])
                row[f"baseline_{col}"] = float(wide.loc[baseline, col])
                row[f"delta_{col}"] = float(
                    wide.loc["two_component", col] - wide.loc[baseline, col]
                )
        rows.append(row)
    return pd.DataFrame(rows)


def species_metric_summary(species_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        species_metrics.groupby("model", observed=True)[
            ["auroc", "auprc", "calibration_error"]
        ]
        .mean()
        .reset_index()
        .sort_values("model")
    )


def species_delta_summary(species_delta: pd.DataFrame) -> dict:
    delta = species_delta["delta_two_component_vs_availability"]
    return {
        "species_count": int(len(species_delta)),
        "two_component_improved_vs_availability": int((delta > 0).sum()),
        "two_component_tied_vs_availability": int((delta == 0).sum()),
        "two_component_declined_vs_availability": int((delta < 0).sum()),
        "mean_delta_two_component_vs_availability": float(delta.mean()),
        "median_delta_two_component_vs_availability": float(delta.median()),
        "min_delta_two_component_vs_availability": float(delta.min()),
        "max_delta_two_component_vs_availability": float(delta.max()),
    }


def ranked_species_delta(species_delta: pd.DataFrame, top: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    sorted_delta = species_delta.sort_values(
        "delta_two_component_vs_availability", ascending=False
    )
    return sorted_delta.head(top).copy(), sorted_delta.tail(top).sort_values(
        "delta_two_component_vs_availability"
    ).copy()


def worst_calibration_bins(
    calibration: pd.DataFrame,
    top: int,
    min_pairs: int,
) -> pd.DataFrame:
    if calibration.empty:
        return calibration
    work = calibration.loc[calibration["pairs"].ge(min_pairs)].copy()
    if work.empty:
        return pd.DataFrame()
    return work.sort_values("calibration_error", ascending=False).head(top).copy()


def focus_season_diagnostics(focus: pd.DataFrame) -> pd.DataFrame:
    work = focus.copy()
    model_names = [
        col.removesuffix("_calibration_error")
        for col in focus.columns
        if col.endswith("_calibration_error")
    ]
    error_cols = [f"{model}_calibration_error" for model in model_names]
    work["best_model_by_abs_error"] = work[error_cols].idxmin(axis=1).str.replace(
        "_calibration_error", "", regex=False
    )
    if {
        "availability_only_calibration_error",
        "two_component_calibration_error",
    }.issubset(work.columns):
        work["delta_two_component_error_vs_availability"] = (
            work["two_component_calibration_error"]
            - work["availability_only_calibration_error"]
        )
    if {
        "effort_only_calibration_error",
        "two_component_calibration_error",
    }.issubset(work.columns):
        work["delta_two_component_error_vs_effort"] = (
            work["two_component_calibration_error"]
            - work["effort_only_calibration_error"]
        )
    return work.sort_values(
        ["common_name", "season_name"],
        kind="stable",
    )


def focus_summary(focus_diag: pd.DataFrame) -> dict:
    summary = {
        "focus_rows": int(len(focus_diag)),
        "best_model_counts": focus_diag["best_model_by_abs_error"]
        .value_counts()
        .to_dict(),
    }
    if "delta_two_component_error_vs_availability" in focus_diag.columns:
        delta = focus_diag["delta_two_component_error_vs_availability"]
        summary["two_component_focus_season_improved_vs_availability"] = int(
            (delta < 0).sum()
        )
        summary["two_component_focus_season_worse_vs_availability"] = int(
            (delta > 0).sum()
        )
        summary["mean_delta_error_two_component_vs_availability"] = float(
            delta.mean()
        )
    return summary


def stratum_delta_summary(strata_deltas: pd.DataFrame) -> dict:
    if strata_deltas.empty:
        return {"strata_delta_rows": 0}
    summary = {"strata_delta_rows": int(len(strata_deltas))}
    for baseline in ["availability_only", "effort_only", "train_prevalence"]:
        work = strata_deltas.loc[strata_deltas["baseline_model"].eq(baseline)]
        if work.empty:
            continue
        delta = work["delta_micro_auprc"]
        summary[f"{baseline}_strata_count"] = int(len(work))
        summary[f"{baseline}_strata_auprc_improved"] = int((delta > 0).sum())
        summary[f"{baseline}_strata_auprc_declined"] = int((delta < 0).sum())
        summary[f"{baseline}_strata_auprc_min_delta"] = float(delta.min())
        summary[f"{baseline}_strata_auprc_max_delta"] = float(delta.max())
        summary[f"{baseline}_strata_auprc_mean_delta"] = float(delta.mean())
    return summary


def ranked_stratum_deltas(
    strata_deltas: pd.DataFrame,
    baseline: str,
    top: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if strata_deltas.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    work = strata_deltas.loc[strata_deltas["baseline_model"].eq(baseline)].copy()
    if work.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    gains = work.sort_values("delta_micro_auprc", ascending=False).head(top)
    smallest = work.sort_values("delta_micro_auprc", ascending=True).head(top)
    calibration_improvements = work.sort_values("delta_ece", ascending=True).head(top)
    calibration_worsening = work.sort_values("delta_ece", ascending=False).head(top)
    return gains, smallest, calibration_improvements, calibration_worsening


def worst_stratum_calibration(strata_metrics: pd.DataFrame, top: int) -> pd.DataFrame:
    if strata_metrics.empty:
        return strata_metrics
    work = strata_metrics.loc[strata_metrics["model"].eq("two_component")].copy()
    if work.empty:
        return pd.DataFrame()
    return work.sort_values("ece", ascending=False).head(top)


def worst_model_calibration(
    metrics: pd.DataFrame,
    model_name: str,
    top: int,
) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    work = metrics.loc[metrics["model"].eq(model_name)].copy()
    if work.empty:
        return pd.DataFrame()
    return work.sort_values("ece", ascending=False).head(top)


def worst_calibration_rows(
    calibration: pd.DataFrame,
    top: int,
    min_pairs: int,
    model_name: str | None = None,
) -> pd.DataFrame:
    if calibration.empty:
        return calibration
    work = calibration.copy()
    if model_name is not None:
        work = work.loc[work["model"].eq(model_name)].copy()
    work = work.loc[work["pairs"].ge(min_pairs)].copy()
    if work.empty:
        return pd.DataFrame()
    return work.sort_values("calibration_error", ascending=False).head(top)


def write_outputs(
    output_dir: Path,
    run_name: str,
    tables: dict[str, pd.DataFrame],
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{run_name}_{name}.csv", index=False)
    (output_dir / f"{run_name}_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir / DEFAULT_OUTPUT_DIR_NAME
    outputs = load_outputs(model_dir, args.run_name)

    metric_deltas = metric_delta(outputs["metrics"])
    species_means = species_metric_summary(outputs["species_metrics"])
    top_gains, top_losses = ranked_species_delta(outputs["species_delta"], args.top)
    calibration_worst = worst_calibration_bins(
        outputs["calibration"], args.top, args.min_bin_pairs
    )
    focus_diag = focus_season_diagnostics(outputs["focus"])
    focus_improvements = focus_diag.sort_values(
        "delta_two_component_error_vs_availability", ascending=True
    ).head(args.top)
    focus_worsening = focus_diag.sort_values(
        "delta_two_component_error_vs_availability", ascending=False
    ).head(args.top)
    (
        stratum_gains,
        stratum_smallest_gains,
        stratum_calibration_improvements,
        stratum_calibration_worsening,
    ) = ranked_stratum_deltas(outputs["strata_deltas"], "availability_only", args.top)
    stratum_calibration_worst = worst_stratum_calibration(
        outputs["strata_metrics"], args.top
    )
    (
        county_season_gains,
        county_season_smallest_gains,
        county_season_calibration_improvements,
        county_season_calibration_worsening,
    ) = ranked_stratum_deltas(
        outputs["county_season_deltas"], "availability_only", args.top
    )
    county_season_calibration_worst = worst_model_calibration(
        outputs["county_season_metrics"], "two_component", args.top
    )
    county_season_worst_bins = worst_calibration_rows(
        outputs["county_season_calibration"],
        args.top,
        args.min_bin_pairs,
        model_name="two_component",
    )
    focus_season_worst_bins = worst_calibration_rows(
        outputs["focus_season_calibration"],
        args.top,
        args.min_bin_pairs,
        model_name="two_component",
    )

    summary = {
        "model_dir": str(model_dir),
        "run_name": args.run_name,
        "min_bin_pairs": int(args.min_bin_pairs),
        "metric_deltas": metric_deltas.to_dict(orient="records"),
        "species_delta_summary": species_delta_summary(outputs["species_delta"]),
        "focus_summary": focus_summary(focus_diag),
        "stratum_delta_summary": stratum_delta_summary(outputs["strata_deltas"]),
        "county_season_delta_summary": stratum_delta_summary(
            outputs["county_season_deltas"]
        ),
    }
    tables = {
        "metric_deltas": metric_deltas,
        "species_metric_means": species_means,
        "top_species_gains_vs_availability": top_gains,
        "top_species_losses_vs_availability": top_losses,
        "worst_calibration_bins": calibration_worst,
        "focus_species_season_diagnostics": focus_diag,
        "focus_species_largest_error_improvements_vs_availability": focus_improvements,
        "focus_species_largest_error_worsening_vs_availability": focus_worsening,
        "stratum_largest_auprc_gains_vs_availability": stratum_gains,
        "stratum_smallest_auprc_gains_vs_availability": stratum_smallest_gains,
        "stratum_largest_ece_improvements_vs_availability": stratum_calibration_improvements,
        "stratum_largest_ece_worsening_vs_availability": stratum_calibration_worsening,
        "stratum_worst_two_component_calibration": stratum_calibration_worst,
        "county_season_largest_auprc_gains_vs_availability": county_season_gains,
        "county_season_smallest_auprc_gains_vs_availability": county_season_smallest_gains,
        "county_season_largest_ece_improvements_vs_availability": county_season_calibration_improvements,
        "county_season_largest_ece_worsening_vs_availability": county_season_calibration_worsening,
        "county_season_worst_two_component_calibration": county_season_calibration_worst,
        "county_season_worst_two_component_calibration_bins": county_season_worst_bins,
        "focus_species_season_worst_two_component_calibration_bins": focus_season_worst_bins,
    }
    write_outputs(output_dir, args.run_name, tables, summary)

    print("Two-component diagnostic summary:")
    print()
    print("Overall deltas, two_component minus baseline:")
    print(metric_deltas.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print()
    print("Mean species-level metrics:")
    print(species_means.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print()
    species_summary = summary["species_delta_summary"]
    print(
        "Species AUPRC delta vs availability-only: "
        f"improved={species_summary['two_component_improved_vs_availability']}, "
        f"declined={species_summary['two_component_declined_vs_availability']}, "
        f"mean_delta={species_summary['mean_delta_two_component_vs_availability']:.5f}"
    )
    print()
    print("Largest species AUPRC gains vs availability-only:")
    cols = [
        "common_name",
        "availability_only",
        "two_component",
        "delta_two_component_vs_availability",
    ]
    print(top_gains[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print()
    print("Largest species AUPRC losses vs availability-only:")
    print(top_losses[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if not calibration_worst.empty:
        print()
        print(
            "Worst predicted-probability calibration bins "
            f"(pairs >= {args.min_bin_pairs}):"
        )
        print(
            calibration_worst.to_string(
                index=False, float_format=lambda x: f"{x:.5f}"
            )
        )
    print()
    print("Focus species/season rows where two_component most improved abs error vs availability:")
    focus_cols = [
        "common_name",
        "season_name",
        "observed_detection_rate",
        "availability_only_predicted_detection_rate",
        "two_component_predicted_detection_rate",
        "delta_two_component_error_vs_availability",
        "best_model_by_abs_error",
    ]
    print(
        focus_improvements[focus_cols].to_string(
            index=False, float_format=lambda x: f"{x:.5f}"
        )
    )
    print()
    print("Focus species/season rows where two_component most worsened abs error vs availability:")
    print(
        focus_worsening[focus_cols].to_string(
            index=False, float_format=lambda x: f"{x:.5f}"
        )
    )
    if not outputs["strata_deltas"].empty:
        print()
        stratum_summary = summary["stratum_delta_summary"]
        print(
            "Stratum AUPRC delta vs availability-only: "
            f"improved={stratum_summary['availability_only_strata_auprc_improved']}, "
            f"declined={stratum_summary['availability_only_strata_auprc_declined']}, "
            f"min_delta={stratum_summary['availability_only_strata_auprc_min_delta']:.5f}, "
            f"max_delta={stratum_summary['availability_only_strata_auprc_max_delta']:.5f}"
        )
        stratum_cols = [
            "stratum_type",
            "stratum",
            "checklists",
            "observed_detection_rate",
            "delta_micro_auprc",
            "delta_ece",
            "delta_calibration_error",
        ]
        print()
        print("Largest stratum AUPRC gains vs availability-only:")
        print(
            stratum_gains[stratum_cols].to_string(
                index=False, float_format=lambda x: f"{x:.5f}"
            )
        )
        print()
        if (stratum_smallest_gains["delta_micro_auprc"] < 0.0).any():
            label = "Largest stratum AUPRC losses vs availability-only:"
        else:
            label = "Smallest stratum AUPRC gains vs availability-only:"
        print(label)
        print(
            stratum_smallest_gains[stratum_cols].to_string(
                index=False, float_format=lambda x: f"{x:.5f}"
            )
        )
        print()
        print("Largest stratum ECE improvements vs availability-only:")
        print(
            stratum_calibration_improvements[stratum_cols].to_string(
                index=False, float_format=lambda x: f"{x:.5f}"
            )
        )
        print()
        print("Largest stratum ECE worsening vs availability-only:")
        print(
            stratum_calibration_worsening[stratum_cols].to_string(
                index=False, float_format=lambda x: f"{x:.5f}"
            )
        )
        if not stratum_calibration_worst.empty:
            print()
            print("Worst two-component stratum calibration:")
            calibration_cols = [
                "stratum_type",
                "stratum",
                "checklists",
                "observed_detection_rate",
                "mean_predicted_detection_rate",
                "ece",
                "max_bin_error",
                "micro_auprc",
            ]
            print(
                stratum_calibration_worst[calibration_cols].to_string(
                    index=False, float_format=lambda x: f"{x:.5f}"
                )
            )
    if not outputs["county_season_deltas"].empty:
        print()
        county_season_summary = summary["county_season_delta_summary"]
        print(
            "County-season AUPRC delta vs availability-only: "
            f"improved={county_season_summary['availability_only_strata_auprc_improved']}, "
            f"declined={county_season_summary['availability_only_strata_auprc_declined']}, "
            f"min_delta={county_season_summary['availability_only_strata_auprc_min_delta']:.5f}, "
            f"max_delta={county_season_summary['availability_only_strata_auprc_max_delta']:.5f}"
        )
        county_season_delta_cols = [
            "stratum",
            "checklists",
            "observed_detection_rate",
            "delta_micro_auprc",
            "delta_ece",
            "delta_calibration_error",
        ]
        print()
        print("Largest county-season AUPRC gains vs availability-only:")
        print(
            county_season_gains[county_season_delta_cols].to_string(
                index=False, float_format=lambda x: f"{x:.5f}"
            )
        )
        print()
        print("Largest county-season ECE worsening vs availability-only:")
        print(
            county_season_calibration_worsening[county_season_delta_cols].to_string(
                index=False, float_format=lambda x: f"{x:.5f}"
            )
        )
    if not county_season_calibration_worst.empty:
        print()
        print("Worst two-component county-season calibration:")
        county_season_metric_cols = [
            "county",
            "season_name",
            "checklists",
            "observed_detection_rate",
            "mean_predicted_detection_rate",
            "ece",
            "max_bin_error",
            "micro_auprc",
        ]
        print(
            county_season_calibration_worst[county_season_metric_cols].to_string(
                index=False, float_format=lambda x: f"{x:.5f}"
            )
        )
    if not county_season_worst_bins.empty:
        print()
        print(
            "Worst two-component county-season probability bins "
            f"(pairs >= {args.min_bin_pairs}):"
        )
        county_season_bin_cols = [
            "county",
            "season_name",
            "bin",
            "pairs",
            "mean_predicted",
            "observed_rate",
            "calibration_error",
        ]
        print(
            county_season_worst_bins[county_season_bin_cols].to_string(
                index=False, float_format=lambda x: f"{x:.5f}"
            )
        )
    if not focus_season_worst_bins.empty:
        print()
        print(
            "Worst two-component focus-species season probability bins "
            f"(pairs >= {args.min_bin_pairs}):"
        )
        focus_bin_cols = [
            "common_name",
            "season_name",
            "bin",
            "pairs",
            "mean_predicted",
            "observed_rate",
            "calibration_error",
        ]
        print(
            focus_season_worst_bins[focus_bin_cols].to_string(
                index=False, float_format=lambda x: f"{x:.5f}"
            )
        )
    print()
    print(f"Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
