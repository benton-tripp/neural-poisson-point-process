"""
Summarize eBird spatial-GNN validation results across split regimes.

This script does not train models. It reads existing spatial-GNN summary JSONs
and optional graph-vs-tabular species comparison CSVs, then writes one compact
CSV for comparing how the same modeling framework behaves under different
spatial validation regimes.

Example:

    python exp/summarize_ebird_split_regime_validation.py --case "primary10x10|data/ebird/graph_top100_spatial_10x10|spatial_gcn_frozen_access_h64_l2_z64|data/ebird/baselines_10x10|both" --case "coastalstress|data/ebird/graph_top100_spatial_10x10_coastalstress|spatial_gcn_frozen_access_h64_l2_z64|data/ebird/baselines/coastalstress|both"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_OUTPUT = (
    "data/ebird/split_diagnostics/spatial_gnn_split_regime_validation.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize spatial-GNN validation across split regimes."
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        metavar="LABEL|GRAPH_DIR|RUN_NAME|BASELINE_DIR|FEATURE_SET",
        help=(
            "Validation case. Repeat for each split regime. BASELINE_DIR is "
            "optional but recommended for tabular aggregate metrics. FEATURE_SET "
            "is optional and defaults to --feature-set."
        ),
    )
    parser.add_argument(
        "--top-species",
        type=int,
        default=100,
        help="Top-N species prefix for tabular baseline summaries. Defaults to 100.",
    )
    parser.add_argument(
        "--feature-set",
        default="both",
        help="Tabular feature-set name. Defaults to both.",
    )
    parser.add_argument(
        "--tabular-model",
        default="mlp",
        choices=["linear", "mlp"],
        help="Tabular model suffix. Defaults to mlp.",
    )
    parser.add_argument(
        "--split",
        default="spatial-stratified",
        help="Tabular split suffix. Defaults to spatial-stratified.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV. Defaults to {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def tabular_model_key(model: str, feature_set: str) -> str:
    return f"{model}_{feature_set}"


def tabular_summary_path(
    baseline_dir: Path,
    top_species: int,
    feature_set: str,
    model: str,
    split: str,
) -> Path:
    model_suffix = "" if model == "linear" else f"_{model}"
    split_suffix = "" if split == "temporal" else f"_{split}"
    return (
        baseline_dir
        / f"top{top_species}_{feature_set}{model_suffix}{split_suffix}_summary.json"
    )


def parse_case(value: str) -> tuple[str, Path, str, Path | None, str | None]:
    parts = [part.strip() for part in value.split("|")]
    if len(parts) not in {3, 4, 5}:
        raise ValueError(
            "Each --case must be LABEL|GRAPH_DIR|RUN_NAME or "
            "LABEL|GRAPH_DIR|RUN_NAME|BASELINE_DIR|FEATURE_SET."
        )
    label = parts[0]
    graph_dir = Path(parts[1])
    run_name = parts[2]
    baseline_dir = Path(parts[3]) if len(parts) == 4 and parts[3] else None
    if len(parts) == 5:
        baseline_dir = Path(parts[3]) if parts[3] else None
    feature_set = parts[4] if len(parts) == 5 and parts[4] else None
    if not label:
        raise ValueError(f"Case has empty label: {value}")
    return label, graph_dir, run_name, baseline_dir, feature_set


def load_tabular_summary(
    baseline_dir: Path | None,
    top_species: int,
    feature_set: str,
    model: str,
    split: str,
) -> dict[str, float | str]:
    if baseline_dir is None:
        return {
            "tabular_summary_path": "",
            "tabular_micro_auroc": np.nan,
            "tabular_micro_auprc": np.nan,
            "tabular_macro_auroc": np.nan,
            "tabular_macro_auprc": np.nan,
            "tabular_ece": np.nan,
        }
    path = tabular_summary_path(baseline_dir, top_species, feature_set, model, split)
    if not path.exists():
        return {
            "tabular_summary_path": str(path),
            "tabular_micro_auroc": np.nan,
            "tabular_micro_auprc": np.nan,
            "tabular_macro_auroc": np.nan,
            "tabular_macro_auprc": np.nan,
            "tabular_ece": np.nan,
        }
    summary = read_json(path)
    key = tabular_model_key(model, feature_set)
    metrics = summary.get(key, {})
    calibration = summary.get("calibration", {}).get(key, {})
    return {
        "tabular_summary_path": str(path),
        "tabular_micro_auroc": metrics.get("micro_auroc", np.nan),
        "tabular_micro_auprc": metrics.get("micro_auprc", np.nan),
        "tabular_macro_auroc": metrics.get("macro_auroc", np.nan),
        "tabular_macro_auprc": metrics.get("macro_auprc", np.nan),
        "tabular_ece": calibration.get("expected_calibration_error", np.nan),
    }


def summarize_species_comparison(path: Path) -> dict[str, float | int | str]:
    if not path.exists():
        return {
            "comparison_path": str(path),
            "comparison_found": False,
            "species_compared": 0,
            "mean_delta_auprc": np.nan,
            "median_delta_auprc": np.nan,
            "species_with_auprc_gain": np.nan,
            "species_with_auprc_loss": np.nan,
            "mean_delta_auroc": np.nan,
            "mean_abs_delta_prevalence": np.nan,
        }
    frame = pd.read_csv(path)
    delta_auprc = frame.get("graph_minus_tabular_auprc", pd.Series(dtype=float))
    delta_auroc = frame.get("graph_minus_tabular_auroc", pd.Series(dtype=float))
    delta_prev = frame.get("graph_minus_tabular_prevalence", pd.Series(dtype=float))
    return {
        "comparison_path": str(path),
        "comparison_found": True,
        "species_compared": int(len(frame)),
        "mean_delta_auprc": float(delta_auprc.mean()) if len(delta_auprc) else np.nan,
        "median_delta_auprc": float(delta_auprc.median()) if len(delta_auprc) else np.nan,
        "species_with_auprc_gain": int((delta_auprc > 0).sum())
        if len(delta_auprc)
        else 0,
        "species_with_auprc_loss": int((delta_auprc < 0).sum())
        if len(delta_auprc)
        else 0,
        "mean_delta_auroc": float(delta_auroc.mean()) if len(delta_auroc) else np.nan,
        "mean_abs_delta_prevalence": float(delta_prev.abs().mean())
        if len(delta_prev)
        else np.nan,
    }


def summarize_case(
    label: str,
    graph_dir: Path,
    run_name: str,
    baseline_dir: Path | None,
    feature_set: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    spatial_dir = graph_dir / "spatial_gnn_baselines"
    summary_path = spatial_dir / f"spatial_gnn_{run_name}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing spatial GNN summary: {summary_path}")
    summary = read_json(summary_path)
    metadata_path = graph_dir / "metadata.json"
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    split = metadata.get("split", {})
    model = summary.get("model", {})
    comparison_path = spatial_dir / f"{run_name}_graph_vs_tabular_species.csv"

    tabular_feature_set = feature_set or args.feature_set
    row: dict[str, Any] = {
        "case": label,
        "graph_dir": str(graph_dir),
        "run_name": run_name,
        "summary_path": str(summary_path),
        "test_checklists": summary.get("checklists", np.nan),
        "test_pairs": summary.get("pairs", np.nan),
        "test_positives": summary.get("positives", np.nan),
        "observed_rate": summary.get("observed_rate", np.nan),
        "graph_micro_auroc": summary.get("auroc", np.nan),
        "graph_micro_auprc": summary.get("auprc", np.nan),
        "graph_macro_auroc": summary.get("species_macro_auroc", np.nan),
        "graph_macro_auprc": summary.get("species_macro_auprc", np.nan),
        "graph_ece": summary.get("probability_bin_ece", np.nan),
        "graph_species_calibration_mae": summary.get(
            "species_calibration_mae", np.nan
        ),
        "blocks_per_dim": split.get("spatial_blocks_per_dim", np.nan),
        "test_block_ids": " ".join(
            str(block) for block in split.get("test_blocks_ids", [])
        ),
        "test_fraction_actual": split.get("test_fraction_actual", np.nan),
        "feature_set": metadata.get("feature_set", ""),
        "tabular_feature_set": tabular_feature_set,
        "component_mode": model.get("component_mode", ""),
        "spatial_channel_mode": model.get("spatial_channel_mode", ""),
        "frozen_access_embeddings": bool(model.get("frozen_access_embeddings")),
        "support_aware_residual": model.get("support_aware_residual", "none"),
    }
    row.update(
        load_tabular_summary(
            baseline_dir,
            args.top_species,
            tabular_feature_set,
            args.tabular_model,
            args.split,
        )
    )
    row["delta_micro_auprc"] = row["graph_micro_auprc"] - row["tabular_micro_auprc"]
    row["delta_macro_auprc"] = row["graph_macro_auprc"] - row["tabular_macro_auprc"]
    row["delta_ece"] = row["graph_ece"] - row["tabular_ece"]
    row.update(summarize_species_comparison(comparison_path))
    return row


def main() -> None:
    args = parse_args()
    if not args.case:
        raise ValueError("Provide at least one --case.")

    rows = []
    for case_text in args.case:
        rows.append(summarize_case(*parse_case(case_text), args=args))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)

    display_columns = [
        "case",
        "graph_micro_auprc",
        "graph_macro_auprc",
        "graph_ece",
        "tabular_micro_auprc",
        "tabular_macro_auprc",
        "delta_micro_auprc",
        "delta_macro_auprc",
        "species_with_auprc_gain",
        "species_with_auprc_loss",
        "test_block_ids",
    ]
    print(f"Wrote split-regime validation summary to {output}")
    print()
    print(frame[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
