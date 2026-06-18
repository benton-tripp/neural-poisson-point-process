"""Score candidate spatial-stratified eBird train/test splits.

Run from the project root:

    python exp/diagnose_ebird_split_candidates.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --spatial-blocks-per-dim 10 --seeds 1-50
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ebird_joint_tabular_baseline import (
    COASTAL_DISTANCE_THRESHOLD_M,
    DEFAULT_PROCESSED_DIR,
    NEAR_WATER_DISTANCE_THRESHOLD_M,
    assign_spatial_blocks,
    build_labels,
    load_inputs,
    select_spatial_test_blocks,
    spatial_stratification_frame,
)


DEFAULT_OUTPUT = (
    "data/ebird/split_diagnostics/top100_spatial_split_candidates.csv"
)


def parse_int_values(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return sorted(set(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate spatial-stratified split seeds and grid sizes "
            "before rebuilding graph datasets."
        )
    )
    parser.add_argument("--processed-dir", default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--top-species", type=int, default=100)
    parser.add_argument("--max-checklists", type=int, default=None)
    parser.add_argument(
        "--spatial-blocks-per-dim",
        default="10",
        help="Comma/range list of grid sizes, e.g. 8,10,12 or 8-12.",
    )
    parser.add_argument(
        "--seeds",
        default="1-50",
        help="Comma/range list of split seeds, e.g. 19,37,101 or 1-50.",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--stratify-species-count", type=int, default=20)
    parser.add_argument(
        "--mode",
        choices=["greedy", "exhaustive"],
        default="greedy",
        help=(
            "greedy replays the current split selector for each seed; exhaustive "
            "scores block combinations directly. Defaults to greedy."
        ),
    )
    parser.add_argument(
        "--min-test-blocks",
        type=int,
        default=None,
        help="Minimum held-out block count for exhaustive mode. Defaults to the greedy count.",
    )
    parser.add_argument(
        "--max-test-blocks",
        type=int,
        default=None,
        help="Maximum held-out block count for exhaustive mode. Defaults to the greedy count.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=200_000,
        help="Maximum exhaustive block combinations to score per grid size. Defaults to 200000.",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=1_000_000,
        help="Maximum exhaustive block combinations to inspect per grid size. Defaults to 1000000.",
    )
    parser.add_argument(
        "--size-tolerance",
        type=float,
        default=0.08,
        help="Allowed absolute deviation from target test fraction in exhaustive mode. Defaults to 0.08.",
    )
    parser.add_argument("--coastal-threshold-m", type=float, default=COASTAL_DISTANCE_THRESHOLD_M)
    parser.add_argument("--near-water-threshold-m", type=float, default=NEAR_WATER_DISTANCE_THRESHOLD_M)
    parser.add_argument(
        "--coastal-block-rate",
        type=float,
        default=0.5,
        help="Minimum block-level coastal checklist rate for counting a test block as coastal.",
    )
    parser.add_argument(
        "--near-water-block-rate",
        type=float,
        default=0.5,
        help="Minimum block-level near-water checklist rate for counting a test block as near-water.",
    )
    parser.add_argument(
        "--min-coastal-test-blocks",
        type=int,
        default=2,
        help="Desired number of coastal held-out blocks. Used as a score penalty only.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"CSV output path. Defaults to {DEFAULT_OUTPUT}.",
    )
    parser.add_argument("--top", type=int, default=20, help="Rows to print.")
    return parser.parse_args()


def rate(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=bool)
    if array.size == 0:
        return float("nan")
    return float(array.mean())


def mean_or_nan(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return float("nan")
    return float(np.nanmean(array))


def split_info_from_blocks(
    block_ids: np.ndarray,
    values: np.ndarray,
    row_std: np.ndarray,
    target_mean: np.ndarray,
    selected_blocks: np.ndarray,
) -> tuple[np.ndarray, dict]:
    test_mask = np.isin(block_ids, selected_blocks)
    selected_values = values[test_mask]
    selected_mean = selected_values.mean(axis=0)
    balance_error = float(np.mean(np.abs((selected_mean - target_mean) / row_std)))
    return test_mask, {
        "test_blocks": int(len(selected_blocks)),
        "test_fraction_actual": float(test_mask.mean()),
        "test_blocks_ids": [int(block) for block in selected_blocks],
        "mean_absolute_standardized_balance_error": balance_error,
    }


def summarize_candidate(
    checklists: pd.DataFrame,
    labels: np.ndarray,
    block_ids: np.ndarray,
    blocks_per_dim: int,
    seed: int,
    test_mask: np.ndarray,
    split_info: dict,
    args: argparse.Namespace,
) -> dict:
    train_mask = ~test_mask
    selected_blocks = np.array(split_info["test_blocks_ids"], dtype=np.int64)

    coastal = checklists["distance_to_coastline_m"].to_numpy(dtype=np.float64) <= args.coastal_threshold_m
    near_water = checklists["distance_to_waterbody_m"].to_numpy(dtype=np.float64) <= args.near_water_threshold_m
    duration = np.log1p(checklists["duration_minutes"].astype(float).to_numpy())
    distance = np.log1p(checklists["effort_distance_km"].fillna(0.0).astype(float).to_numpy())
    observers = np.log1p(checklists["number_observers"].astype(float).to_numpy())
    traveling = (checklists["protocol_code"].to_numpy() == "P22")

    block_rows = []
    for block in selected_blocks:
        mask = block_ids == block
        block_rows.append(
            {
                "block": int(block),
                "checklists": int(mask.sum()),
                "coastal_rate": rate(coastal[mask]),
                "near_water_rate": rate(near_water[mask]),
            }
        )
    block_frame = pd.DataFrame(block_rows)
    coastal_blocks = int((block_frame["coastal_rate"] >= args.coastal_block_rate).sum())
    near_water_blocks = int(
        (block_frame["near_water_rate"] >= args.near_water_block_rate).sum()
    )

    species_count = min(args.stratify_species_count, labels.shape[1])
    train_prev = labels[train_mask, :species_count].mean(axis=0)
    test_prev = labels[test_mask, :species_count].mean(axis=0)
    species_prevalence_mae = float(np.mean(np.abs(test_prev - train_prev)))

    coastal_penalty = max(0, args.min_coastal_test_blocks - coastal_blocks) * 0.25
    size_penalty = abs(split_info["test_fraction_actual"] - args.test_fraction) * 2.0
    score = (
        split_info["mean_absolute_standardized_balance_error"]
        + species_prevalence_mae
        + coastal_penalty
        + size_penalty
    )

    return {
        "score": float(score),
        "blocks_per_dim": int(blocks_per_dim),
        "seed": int(seed),
        "test_blocks": split_info["test_blocks"],
        "test_block_ids": " ".join(str(block) for block in selected_blocks),
        "test_fraction_actual": split_info["test_fraction_actual"],
        "balance_error": split_info["mean_absolute_standardized_balance_error"],
        "species_prevalence_mae": species_prevalence_mae,
        "test_checklists": int(test_mask.sum()),
        "coastal_test_blocks": coastal_blocks,
        "near_water_test_blocks": near_water_blocks,
        "test_coastal_rate": rate(coastal[test_mask]),
        "train_coastal_rate": rate(coastal[train_mask]),
        "test_near_water_rate": rate(near_water[test_mask]),
        "train_near_water_rate": rate(near_water[train_mask]),
        "test_traveling_rate": rate(traveling[test_mask]),
        "train_traveling_rate": rate(traveling[train_mask]),
        "test_duration_log1p_mean": mean_or_nan(duration[test_mask]),
        "train_duration_log1p_mean": mean_or_nan(duration[train_mask]),
        "test_distance_log1p_mean": mean_or_nan(distance[test_mask]),
        "train_distance_log1p_mean": mean_or_nan(distance[train_mask]),
        "test_observers_log1p_mean": mean_or_nan(observers[test_mask]),
        "train_observers_log1p_mean": mean_or_nan(observers[train_mask]),
    }


def build_block_summary(
    checklists: pd.DataFrame,
    labels: np.ndarray,
    block_ids: np.ndarray,
    stratify_values: pd.DataFrame,
    args: argparse.Namespace,
) -> dict:
    unique_blocks = np.array(sorted(np.unique(block_ids)), dtype=np.int64)
    block_lookup = {int(block): idx for idx, block in enumerate(unique_blocks)}
    block_index = np.array([block_lookup[int(block)] for block in block_ids], dtype=np.int64)
    block_count = np.bincount(block_index, minlength=len(unique_blocks)).astype(np.float64)

    values = stratify_values.to_numpy(dtype=np.float64)
    feature_sums = np.zeros((len(unique_blocks), values.shape[1]), dtype=np.float64)
    np.add.at(feature_sums, block_index, values)

    species_count = min(args.stratify_species_count, labels.shape[1])
    species_sums = np.zeros((len(unique_blocks), species_count), dtype=np.float64)
    np.add.at(species_sums, block_index, labels[:, :species_count].astype(np.float64))

    coastal = (
        checklists["distance_to_coastline_m"].to_numpy(dtype=np.float64)
        <= args.coastal_threshold_m
    ).astype(np.float64)
    near_water = (
        checklists["distance_to_waterbody_m"].to_numpy(dtype=np.float64)
        <= args.near_water_threshold_m
    ).astype(np.float64)
    traveling = (checklists["protocol_code"].to_numpy() == "P22").astype(np.float64)
    duration = np.log1p(checklists["duration_minutes"].astype(float).to_numpy())
    distance = np.log1p(
        checklists["effort_distance_km"].fillna(0.0).astype(float).to_numpy()
    )
    observers = np.log1p(checklists["number_observers"].astype(float).to_numpy())

    def block_sum(array: np.ndarray) -> np.ndarray:
        output = np.zeros(len(unique_blocks), dtype=np.float64)
        np.add.at(output, block_index, array)
        return output

    row_std = values.std(axis=0)
    row_std[row_std == 0.0] = 1.0
    return {
        "unique_blocks": unique_blocks,
        "block_count": block_count,
        "feature_sums": feature_sums,
        "species_sums": species_sums,
        "coastal_sum": block_sum(coastal),
        "near_water_sum": block_sum(near_water),
        "traveling_sum": block_sum(traveling),
        "duration_sum": block_sum(duration),
        "distance_sum": block_sum(distance),
        "observers_sum": block_sum(observers),
        "total_count": float(len(checklists)),
        "total_feature_sum": values.sum(axis=0),
        "total_species_sum": labels[:, :species_count].astype(np.float64).sum(axis=0),
        "total_coastal_sum": float(coastal.sum()),
        "total_near_water_sum": float(near_water.sum()),
        "total_traveling_sum": float(traveling.sum()),
        "total_duration_sum": float(duration.sum()),
        "total_distance_sum": float(distance.sum()),
        "total_observers_sum": float(observers.sum()),
        "target_mean": values.mean(axis=0),
        "row_std": row_std,
    }


def summarize_block_candidate(
    block_summary: dict,
    selected_positions: np.ndarray,
    blocks_per_dim: int,
    args: argparse.Namespace,
    seed: int = -1,
) -> dict:
    block_count = block_summary["block_count"]
    selected_count = float(block_count[selected_positions].sum())
    total_count = float(block_summary["total_count"])
    train_count = total_count - selected_count
    if selected_count <= 0 or train_count <= 0:
        raise ValueError("Candidate split has empty train or test set.")

    selected_feature_sum = block_summary["feature_sums"][selected_positions].sum(axis=0)
    selected_mean = selected_feature_sum / selected_count
    balance_error = float(
        np.mean(
            np.abs(
                (selected_mean - block_summary["target_mean"])
                / block_summary["row_std"]
            )
        )
    )

    selected_species_sum = block_summary["species_sums"][selected_positions].sum(axis=0)
    train_species_sum = block_summary["total_species_sum"] - selected_species_sum
    species_prevalence_mae = float(
        np.mean(np.abs(selected_species_sum / selected_count - train_species_sum / train_count))
    )

    selected_coastal = float(block_summary["coastal_sum"][selected_positions].sum())
    selected_near_water = float(block_summary["near_water_sum"][selected_positions].sum())
    selected_traveling = float(block_summary["traveling_sum"][selected_positions].sum())
    selected_duration = float(block_summary["duration_sum"][selected_positions].sum())
    selected_distance = float(block_summary["distance_sum"][selected_positions].sum())
    selected_observers = float(block_summary["observers_sum"][selected_positions].sum())

    block_coastal_rates = (
        block_summary["coastal_sum"][selected_positions]
        / block_summary["block_count"][selected_positions]
    )
    block_near_water_rates = (
        block_summary["near_water_sum"][selected_positions]
        / block_summary["block_count"][selected_positions]
    )
    coastal_blocks = int((block_coastal_rates >= args.coastal_block_rate).sum())
    near_water_blocks = int((block_near_water_rates >= args.near_water_block_rate).sum())

    split_info = {
        "test_blocks": int(len(selected_positions)),
        "test_fraction_actual": float(selected_count / total_count),
        "test_blocks_ids": [
            int(block)
            for block in block_summary["unique_blocks"][selected_positions]
        ],
        "mean_absolute_standardized_balance_error": balance_error,
    }
    coastal_penalty = max(0, args.min_coastal_test_blocks - coastal_blocks) * 0.25
    size_penalty = abs(split_info["test_fraction_actual"] - args.test_fraction) * 2.0
    score = balance_error + species_prevalence_mae + coastal_penalty + size_penalty

    train_coastal = block_summary["total_coastal_sum"] - selected_coastal
    train_near_water = block_summary["total_near_water_sum"] - selected_near_water
    train_traveling = block_summary["total_traveling_sum"] - selected_traveling
    train_duration = block_summary["total_duration_sum"] - selected_duration
    train_distance = block_summary["total_distance_sum"] - selected_distance
    train_observers = block_summary["total_observers_sum"] - selected_observers

    return {
        "score": float(score),
        "blocks_per_dim": int(blocks_per_dim),
        "seed": int(seed),
        "test_blocks": split_info["test_blocks"],
        "test_block_ids": " ".join(str(block) for block in split_info["test_blocks_ids"]),
        "test_fraction_actual": split_info["test_fraction_actual"],
        "balance_error": balance_error,
        "species_prevalence_mae": species_prevalence_mae,
        "test_checklists": int(selected_count),
        "coastal_test_blocks": coastal_blocks,
        "near_water_test_blocks": near_water_blocks,
        "test_coastal_rate": float(selected_coastal / selected_count),
        "train_coastal_rate": float(train_coastal / train_count),
        "test_near_water_rate": float(selected_near_water / selected_count),
        "train_near_water_rate": float(train_near_water / train_count),
        "test_traveling_rate": float(selected_traveling / selected_count),
        "train_traveling_rate": float(train_traveling / train_count),
        "test_duration_log1p_mean": float(selected_duration / selected_count),
        "train_duration_log1p_mean": float(train_duration / train_count),
        "test_distance_log1p_mean": float(selected_distance / selected_count),
        "train_distance_log1p_mean": float(train_distance / train_count),
        "test_observers_log1p_mean": float(selected_observers / selected_count),
        "train_observers_log1p_mean": float(train_observers / train_count),
    }


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    grids = parse_int_values(args.spatial_blocks_per_dim)
    seeds = parse_int_values(args.seeds)
    checklists, detections, species = load_inputs(
        processed_dir,
        args.top_species,
        args.max_checklists,
    )
    labels = build_labels(checklists, detections, species)

    rows = []
    for blocks_per_dim in grids:
        block_ids = assign_spatial_blocks(checklists, blocks_per_dim)
        stratify_values = spatial_stratification_frame(
            checklists,
            labels,
            species,
            args.stratify_species_count,
        )
        greedy_masks = []
        for seed in seeds:
            test_mask, split_info = select_spatial_test_blocks(
                block_ids,
                stratify_values,
                args.test_fraction,
                seed,
            )
            greedy_masks.append((seed, test_mask, split_info))
            rows.append(
                summarize_candidate(
                    checklists,
                    labels,
                    block_ids,
                    blocks_per_dim,
                    seed,
                    test_mask,
                    split_info,
                    args,
                )
            )
        if args.mode == "greedy":
            continue

        block_summary = build_block_summary(
            checklists,
            labels,
            block_ids,
            stratify_values,
            args,
        )
        values = stratify_values.to_numpy(dtype=np.float64)
        row_std = values.std(axis=0)
        row_std[row_std == 0.0] = 1.0
        target_mean = values.mean(axis=0)
        unique_blocks = np.array(sorted(np.unique(block_ids)), dtype=np.int64)
        block_counts = block_summary["block_count"]
        target_count = len(block_ids) * args.test_fraction

        greedy_block_counts = [
            int(info["test_blocks"]) for _, _, info in greedy_masks
        ]
        min_blocks = args.min_test_blocks or min(greedy_block_counts)
        max_blocks = args.max_test_blocks or max(greedy_block_counts)
        scored = 0
        inspected = 0
        for block_count in range(min_blocks, max_blocks + 1):
            combinations = itertools.combinations(range(len(unique_blocks)), block_count)
            for selected_tuple in combinations:
                inspected += 1
                if inspected > args.max_combinations:
                    break
                selected_positions = np.array(selected_tuple, dtype=np.int64)
                selected_count = float(block_counts[selected_positions].sum())
                if (
                    abs(selected_count - target_count) / len(block_ids)
                    > args.size_tolerance
                ):
                    continue
                rows.append(
                    summarize_block_candidate(
                        block_summary,
                        selected_positions,
                        blocks_per_dim,
                        args,
                    )
                )
                scored += 1
                if scored >= args.max_candidates:
                    break
            if scored >= args.max_candidates or inspected > args.max_combinations:
                break

    results = pd.DataFrame(rows).sort_values(
        ["coastal_test_blocks", "score"],
        ascending=[False, True],
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "processed_dir": str(processed_dir),
                "top_species": args.top_species,
                "spatial_blocks_per_dim": grids,
                "seeds": seeds,
                "mode": args.mode,
                "min_test_blocks": args.min_test_blocks,
                "max_test_blocks": args.max_test_blocks,
                "max_candidates": args.max_candidates,
                "test_fraction": args.test_fraction,
                "stratify_species_count": args.stratify_species_count,
                "coastal_threshold_m": args.coastal_threshold_m,
                "near_water_threshold_m": args.near_water_threshold_m,
                "score": (
                    "balance_error + species_prevalence_mae + coastal block "
                    "penalty + test fraction penalty"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote split candidate diagnostics to {output}")
    print("\nBest candidate splits, prioritizing coastal block coverage:")
    columns = [
        "score",
        "blocks_per_dim",
        "seed",
        "test_blocks",
        "test_fraction_actual",
        "balance_error",
        "species_prevalence_mae",
        "coastal_test_blocks",
        "near_water_test_blocks",
        "test_coastal_rate",
        "train_coastal_rate",
        "test_block_ids",
    ]
    print(results[columns].head(args.top).to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
