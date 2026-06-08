"""
Run a small residual/gated spatial-cell GNN tuning grid.

This is a thin, resumable wrapper around exp/ebird_spatial_gnn_baseline.py.
It is intended for conservative tuning of residual strength/capacity before
moving to a larger heterogeneous graph.

Run from the project root, for example:

    python exp/run_ebird_spatial_gnn_grid.py --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path


DEFAULT_GRAPH_DIR = "data/ebird/graph_top100_spatial"


def parse_csv_values(value: str, cast):
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable grid over residual/gated spatial GNN settings."
    )
    parser.add_argument(
        "--graph-dir",
        default=DEFAULT_GRAPH_DIR,
        help=f"Graph dataset directory. Defaults to {DEFAULT_GRAPH_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Spatial GNN output directory. Defaults to graph-dir/spatial_gnn_baselines.",
    )
    parser.add_argument(
        "--modes",
        default="residual,gated",
        help="Comma-separated GNN modes to run. Defaults to residual,gated.",
    )
    parser.add_argument(
        "--cell-hidden-dims",
        default="32,64",
        help="Comma-separated spatial-cell hidden dims. Defaults to 32,64.",
    )
    parser.add_argument(
        "--cell-layers",
        default="1,2",
        help="Comma-separated spatial-cell layer counts. Defaults to 1,2.",
    )
    parser.add_argument(
        "--weight-decays",
        default="0.0001,0.001",
        help="Comma-separated AdamW weight decays. Defaults to 0.0001,0.001.",
    )
    parser.add_argument(
        "--gate-init-biases",
        default="-2,-3",
        help="Comma-separated gate init biases for gated runs. Defaults to -2,-3.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--spatial-grid-size-m", type=float, default=25_000.0)
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on commands run after filtering/skipping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Run even when the target summary JSON already exists.",
    )
    return parser.parse_args()


def format_float_for_name(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def build_run_name(
    mode: str,
    cell_hidden_dim: int,
    cell_layers: int,
    weight_decay: float,
    gate_init_bias: float | None,
) -> str:
    parts = [
        "spatial_gcn",
        mode,
        "h128",
        "l2",
        "z128",
        f"cell{cell_hidden_dim}",
        f"cl{cell_layers}",
        f"wd{format_float_for_name(weight_decay)}",
    ]
    if mode == "gated" and gate_init_bias is not None:
        parts.append(f"gb{format_float_for_name(gate_init_bias)}")
    return "_".join(parts)


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir) if args.output_dir else graph_dir / "spatial_gnn_baselines"

    modes = parse_csv_values(args.modes, str)
    cell_hidden_dims = parse_csv_values(args.cell_hidden_dims, int)
    cell_layers_values = parse_csv_values(args.cell_layers, int)
    weight_decays = parse_csv_values(args.weight_decays, float)
    gate_init_biases = parse_csv_values(args.gate_init_biases, float)

    commands: list[list[str]] = []
    for mode, cell_hidden_dim, cell_layers, weight_decay in itertools.product(
        modes, cell_hidden_dims, cell_layers_values, weight_decays
    ):
        if mode not in {"residual", "gated", "concat"}:
            raise ValueError(f"Unsupported mode: {mode}")
        gate_values: list[float | None] = gate_init_biases if mode == "gated" else [None]
        for gate_init_bias in gate_values:
            run_name = build_run_name(
                mode, cell_hidden_dim, cell_layers, weight_decay, gate_init_bias
            )
            summary_path = output_dir / f"spatial_gnn_{run_name}_summary.json"
            if summary_path.exists() and not args.overwrite:
                print(f"Skipping existing run: {run_name}")
                continue
            command = [
                sys.executable,
                "exp/ebird_spatial_gnn_baseline.py",
                "--graph-dir",
                str(graph_dir),
                "--output-dir",
                str(output_dir),
                "--run-name",
                run_name,
                "--gnn-mode",
                mode,
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--hidden-dim",
                str(args.hidden_dim),
                "--hidden-layers",
                str(args.hidden_layers),
                "--latent-dim",
                str(args.latent_dim),
                "--cell-hidden-dim",
                str(cell_hidden_dim),
                "--cell-layers",
                str(cell_layers),
                "--dropout",
                str(args.dropout),
                "--learning-rate",
                str(args.learning_rate),
                "--weight-decay",
                str(weight_decay),
                "--spatial-grid-size-m",
                str(args.spatial_grid_size_m),
            ]
            if mode == "gated" and gate_init_bias is not None:
                command.extend(["--gate-init-bias", str(gate_init_bias)])
            commands.append(command)

    if args.max_runs is not None:
        commands = commands[: args.max_runs]

    if not commands:
        print("No runs to execute.")
        return

    for idx, command in enumerate(commands, start=1):
        print(f"\n[{idx}/{len(commands)}] {' '.join(command)}")
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
