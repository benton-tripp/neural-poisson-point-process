"""
Plot diagnostics for spatial-cell GNN runs.

The script reads summary JSONs, calibration CSVs, and optional graph-vs-tabular
species comparison CSVs from the spatial GNN output directory. It writes a
compact run summary CSV plus PNGs that show ranking/calibration tradeoffs and
species-level wins/losses.

Run from the project root:

    python exp/plot_ebird_spatial_gnn_grid.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot spatial GNN grid summary and diagnostics."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--spatial-gnn-dir",
        default=None,
        help="Spatial GNN output directory. Defaults to graph-dir/spatial_gnn_baselines.",
    )
    parser.add_argument(
        "--baseline-dir",
        default="data/ebird/baselines",
        help="Tabular baseline directory. Defaults to data/ebird/baselines.",
    )
    parser.add_argument(
        "--link-baseline-dir",
        default=None,
        help="All-species link baseline directory. Defaults to graph-dir/all_species_link_baselines.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Plot output directory. Defaults to spatial-gnn-dir/diagnostics.",
    )
    parser.add_argument(
        "--top-species",
        type=int,
        default=12,
        help="Species to show in top/bottom delta plots. Defaults to 12.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_label_from_summary(path: Path, summary: dict) -> str:
    if path.name.startswith("spatial_gnn_"):
        return path.name.removeprefix("spatial_gnn_").removesuffix("_summary.json")
    if path.name.startswith("all_species_link_"):
        return path.name.removeprefix("all_species_link_").removesuffix("_summary.json")
    return path.stem.removesuffix("_summary")


def flatten_graph_summary(path: Path, model_family: str) -> dict:
    summary = load_json(path)
    model = summary.get("model", {})
    return {
        "run": run_label_from_summary(path, summary),
        "model_family": model_family,
        "architecture": model.get("architecture", model_family),
        "gnn_mode": model.get("gnn_mode", ""),
        "feature_augmentation": model.get("feature_augmentation", ""),
        "spatial_residual": model.get("spatial_residual", ""),
        "cell_hidden_dim": model.get("cell_hidden_dim", np.nan),
        "cell_layers": model.get("cell_layers", np.nan),
        "gate_init_bias": model.get("gate_init_bias", np.nan),
        "weight_decay": model.get("weight_decay", np.nan),
        "micro_auroc": summary.get("auroc", np.nan),
        "micro_auprc": summary.get("auprc", np.nan),
        "macro_auroc": summary.get("species_macro_auroc", np.nan),
        "macro_auprc": summary.get("species_macro_auprc", np.nan),
        "ece": summary.get("probability_bin_ece", np.nan),
        "max_bin_error": summary.get("probability_bin_max_error", np.nan),
        "species_calibration_mae": summary.get("species_calibration_mae", np.nan),
        "path": str(path),
    }


def flatten_tabular_summary(path: Path) -> list[dict]:
    summary = load_json(path)
    rows = []
    for key, values in summary.items():
        if not isinstance(values, dict):
            continue
        if not {"micro_auroc", "micro_auprc", "macro_auroc", "macro_auprc"}.issubset(
            values
        ):
            continue
        calibration = summary.get("calibration", {}).get(key, {})
        rows.append(
            {
                "run": key,
                "model_family": "tabular",
                "architecture": key,
                "gnn_mode": "",
                "feature_augmentation": "",
                "spatial_residual": "",
                "cell_hidden_dim": np.nan,
                "cell_layers": np.nan,
                "gate_init_bias": np.nan,
                "weight_decay": np.nan,
                "micro_auroc": values.get("micro_auroc", np.nan),
                "micro_auprc": values.get("micro_auprc", np.nan),
                "macro_auroc": values.get("macro_auroc", np.nan),
                "macro_auprc": values.get("macro_auprc", np.nan),
                "ece": calibration.get("expected_calibration_error", np.nan),
                "max_bin_error": calibration.get("max_probability_bin_error", np.nan),
                "species_calibration_mae": values.get("species_calibration_mae", np.nan),
                "path": str(path),
            }
        )
    return rows


def collect_summaries(
    spatial_gnn_dir: Path, baseline_dir: Path, link_baseline_dir: Path
) -> pd.DataFrame:
    rows = []
    for path in sorted(spatial_gnn_dir.glob("spatial_gnn_*_summary.json")):
        rows.append(flatten_graph_summary(path, "spatial_gnn"))
    for path in sorted(link_baseline_dir.glob("all_species_link_*_summary.json")):
        rows.append(flatten_graph_summary(path, "link_baseline"))
    tabular_path = baseline_dir / "top100_both_mlp_spatial-stratified_summary.json"
    if tabular_path.exists():
        rows.extend(flatten_tabular_summary(tabular_path))
    if not rows:
        raise FileNotFoundError("No summary JSON files found.")
    df = pd.DataFrame(rows)
    return df.sort_values(["model_family", "micro_auprc"], ascending=[True, False])


def short_label(label: str, max_len: int = 34) -> str:
    return label if len(label) <= max_len else label[: max_len - 1] + "..."


def plot_metric_bars(df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        ("micro_auprc", "Micro AUPRC"),
        ("macro_auprc", "Macro AUPRC"),
        ("ece", "ECE"),
        ("species_calibration_mae", "Species calibration MAE"),
    ]
    plot_df = df.sort_values("micro_auprc", ascending=False).head(16).copy()
    labels = [short_label(run) for run in plot_df["run"]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes_flat = axes.ravel()
    colors = plot_df["model_family"].map(
        {"spatial_gnn": "#2f80ed", "link_baseline": "#8e44ad", "tabular": "#27ae60"}
    ).fillna("#555555")
    for ax, (metric, title) in zip(axes_flat, metrics):
        ax.barh(labels, plot_df[metric], color=colors)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_tradeoff(df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    family_to_marker = {"spatial_gnn": "o", "link_baseline": "s", "tabular": "^"}
    family_to_color = {"spatial_gnn": "#2f80ed", "link_baseline": "#8e44ad", "tabular": "#27ae60"}
    for family, group in df.groupby("model_family"):
        axes[0].scatter(
            group["ece"],
            group["micro_auprc"],
            label=family,
            marker=family_to_marker.get(family, "o"),
            color=family_to_color.get(family, "#555555"),
            s=70,
            alpha=0.85,
        )
        axes[1].scatter(
            group["species_calibration_mae"],
            group["macro_auprc"],
            label=family,
            marker=family_to_marker.get(family, "o"),
            color=family_to_color.get(family, "#555555"),
            s=70,
            alpha=0.85,
        )
    axes[0].set_xlabel("ECE")
    axes[0].set_ylabel("Micro AUPRC")
    axes[0].set_title("Ranking vs probability calibration")
    axes[1].set_xlabel("Species calibration MAE")
    axes[1].set_ylabel("Macro AUPRC")
    axes[1].set_title("Species ranking vs species calibration")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_calibration_curves(spatial_gnn_dir: Path, output_path: Path) -> None:
    files = sorted(spatial_gnn_dir.glob("spatial_gnn_*_test_calibration.csv"))
    if not files:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--", label="perfect")
    for path in files:
        df = pd.read_csv(path)
        df = df[df["calibration_type"] == "predicted_probability_bin"]
        if df.empty:
            continue
        label = path.name.removeprefix("spatial_gnn_").removesuffix("_test_calibration.csv")
        ax.plot(
            df["mean_predicted"],
            df["observed_rate"],
            marker="o",
            linewidth=1.5,
            label=short_label(label, 28),
        )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed detection rate")
    ax.set_title("Spatial GNN calibration curves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{text[:70]}_{digest}"


def plot_species_deltas(spatial_gnn_dir: Path, output_dir: Path, top_species: int) -> None:
    files = sorted(spatial_gnn_dir.glob("*_graph_vs_tabular_species.csv"))
    for path in files:
        df = pd.read_csv(path)
        if "graph_minus_tabular_auprc" not in df.columns:
            continue
        top = df.nlargest(top_species, "graph_minus_tabular_auprc")
        bottom = df.nsmallest(top_species, "graph_minus_tabular_auprc")
        plot_df = pd.concat([bottom, top], ignore_index=True)
        labels = plot_df["graph_common_name"].astype(str)
        colors = np.where(plot_df["graph_minus_tabular_auprc"] >= 0, "#2f80ed", "#c0392b")
        height = max(7, 0.34 * len(plot_df))
        fig, ax = plt.subplots(figsize=(10, height))
        ax.barh(labels, plot_df["graph_minus_tabular_auprc"], color=colors)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Graph minus tabular AUPRC")
        ax.set_title(short_label(path.name.removesuffix(".csv"), 90))
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"species_delta_{safe_filename(path.stem)}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    spatial_gnn_dir = (
        Path(args.spatial_gnn_dir)
        if args.spatial_gnn_dir
        else graph_dir / "spatial_gnn_baselines"
    )
    link_baseline_dir = (
        Path(args.link_baseline_dir)
        if args.link_baseline_dir
        else graph_dir / "all_species_link_baselines"
    )
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output_dir) if args.output_dir else spatial_gnn_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = collect_summaries(spatial_gnn_dir, baseline_dir, link_baseline_dir)
    summary_path = output_dir / "spatial_gnn_run_summary.csv"
    summary.to_csv(summary_path, index=False)
    plot_metric_bars(summary, output_dir / "run_metric_bars.png")
    plot_tradeoff(summary, output_dir / "ranking_calibration_tradeoff.png")
    plot_calibration_curves(spatial_gnn_dir, output_dir / "spatial_gnn_calibration_curves.png")
    plot_species_deltas(spatial_gnn_dir, output_dir, args.top_species)

    print(f"Wrote {summary_path}")
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
