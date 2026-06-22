"""
Diagnose locality-season replication support for occupancy/detection modeling.

Run after building the locality-season dataset:

    python exp/diagnose_ebird_locality_season_replication.py --dataset-dir data/ebird/locality_season_top100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATASET_DIR = "data/ebird/locality_season_top100"
DEFAULT_FOCUS_SPECIES = [
    "Wood Thrush",
    "Red-headed Woodpecker",
    "Green Heron",
    "Great Egret",
    "House Sparrow",
    "Eastern Towhee",
    "Black-and-white Warbler",
    "Double-crested Cormorant",
    "Northern Cardinal",
    "Eastern Meadowlark",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize locality-season replication support by species, season, and effort."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Locality-season dataset directory. Defaults to {DEFAULT_DATASET_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to dataset-dir/diagnostics.",
    )
    parser.add_argument(
        "--focus-species",
        nargs="*",
        default=DEFAULT_FOCUS_SPECIES,
        help="Common names to include in the focus species table.",
    )
    parser.add_argument(
        "--min-positive-locality-seasons",
        type=int,
        default=25,
        help="Threshold used to flag species-season cells with enough positive support. Defaults to 25.",
    )
    return parser.parse_args()


def read_metadata(dataset_dir: Path) -> dict:
    path = dataset_dir / "metadata.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_overall(locality_seasons: pd.DataFrame, triplets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(
        {
            "metric": "locality_seasons",
            "value": len(locality_seasons),
        }
    )
    rows.append(
        {
            "metric": "eligible_locality_seasons",
            "value": int(locality_seasons["eligible_for_species_table"].sum()),
        }
    )
    rows.append(
        {
            "metric": "adequately_sampled_locality_seasons",
            "value": int(locality_seasons["adequate_sampling"].sum()),
        }
    )
    rows.append(
        {
            "metric": "locality_season_species_rows",
            "value": len(triplets),
        }
    )
    rows.append(
        {
            "metric": "species_triplets_with_detection",
            "value": int((triplets["n_detections"] > 0).sum()),
        }
    )
    rows.append(
        {
            "metric": "adequately_sampled_triplets",
            "value": int(triplets["adequate_sampling"].sum()),
        }
    )
    rows.append(
        {
            "metric": "adequately_sampled_positive_triplets",
            "value": int(((triplets["adequate_sampling"]) & (triplets["n_detections"] > 0)).sum()),
        }
    )
    return pd.DataFrame(rows)


def summarize_by_season(locality_seasons: pd.DataFrame, triplets: pd.DataFrame) -> pd.DataFrame:
    season_locality = (
        locality_seasons.groupby("season_name", observed=True)
        .agg(
            locality_seasons=("locality_season_id", "size"),
            eligible_locality_seasons=("eligible_for_species_table", "sum"),
            adequately_sampled_locality_seasons=("adequate_sampling", "sum"),
            checklists=("n_checklists", "sum"),
            median_checklists=("n_checklists", "median"),
            median_dates=("n_dates", "median"),
        )
        .reset_index()
    )
    season_species = (
        triplets.groupby("season_name", observed=True)
        .agg(
            species_triplets=("species_key", "size"),
            positive_triplets=("n_detections", lambda s: int((s > 0).sum())),
            detections=("n_detections", "sum"),
            mean_detection_rate=("naive_detection_rate", "mean"),
        )
        .reset_index()
    )
    return season_locality.merge(season_species, on="season_name", how="left")


def summarize_by_species(triplets: pd.DataFrame) -> pd.DataFrame:
    rows = (
        triplets.groupby(["species_key", "common_name", "scientific_name"], observed=True)
        .agg(
            locality_seasons=("locality_season_id", "size"),
            positive_locality_seasons=("n_detections", lambda s: int((s > 0).sum())),
            adequately_sampled_locality_seasons=("adequate_sampling", "sum"),
            adequately_sampled_positive_locality_seasons=(
                "n_detections",
                lambda s: int(((s > 0) & triplets.loc[s.index, "adequate_sampling"]).sum()),
            ),
            total_checklists=("n_checklists", "sum"),
            total_detections=("n_detections", "sum"),
            median_checklists=("n_checklists", "median"),
            median_positive_detection_rate=(
                "naive_detection_rate",
                lambda s: float(s[s > 0].median()) if (s > 0).any() else np.nan,
            ),
            seasons_with_detection=("season_name", lambda s: triplets.loc[s.index].query("n_detections > 0")["season_name"].nunique()),
        )
        .reset_index()
    )
    rows["positive_locality_season_rate"] = rows["positive_locality_seasons"] / rows[
        "locality_seasons"
    ].clip(lower=1)
    rows["adequate_positive_rate"] = rows[
        "adequately_sampled_positive_locality_seasons"
    ] / rows["adequately_sampled_locality_seasons"].clip(lower=1)
    return rows.sort_values("positive_locality_seasons", ascending=False)


def summarize_species_season(
    triplets: pd.DataFrame, min_positive_locality_seasons: int
) -> pd.DataFrame:
    rows = (
        triplets.groupby(["common_name", "season_name"], observed=True)
        .agg(
            locality_seasons=("locality_season_id", "size"),
            positive_locality_seasons=("n_detections", lambda s: int((s > 0).sum())),
            adequately_sampled_locality_seasons=("adequate_sampling", "sum"),
            detections=("n_detections", "sum"),
            checklists=("n_checklists", "sum"),
            mean_detection_rate=("naive_detection_rate", "mean"),
            positive_median_detection_rate=(
                "naive_detection_rate",
                lambda s: float(s[s > 0].median()) if (s > 0).any() else np.nan,
            ),
        )
        .reset_index()
    )
    rows["positive_support_ok"] = (
        rows["positive_locality_seasons"] >= min_positive_locality_seasons
    )
    return rows.sort_values(
        ["common_name", "positive_locality_seasons"],
        ascending=[True, False],
    )


def summarize_effort_support(locality_seasons: pd.DataFrame) -> pd.DataFrame:
    bins = pd.cut(
        locality_seasons["n_checklists"],
        bins=[0, 2, 4, 9, 19, 49, np.inf],
        labels=["1-2", "3-4", "5-9", "10-19", "20-49", "50+"],
    )
    work = locality_seasons.copy()
    work["checklist_count_bin"] = bins
    return (
        work.groupby("checklist_count_bin", observed=True)
        .agg(
            locality_seasons=("locality_season_id", "size"),
            adequately_sampled_locality_seasons=("adequate_sampling", "sum"),
            median_dates=("n_dates", "median"),
            median_duration_bin_count=("duration_bin_count", "median"),
            median_unique_observers=("unique_observers", "median"),
            median_stationary_rate=("stationary_rate", "median"),
            median_traveling_rate=("traveling_rate", "median"),
        )
        .reset_index()
    )


def write_outputs(
    output_dir: Path,
    overall: pd.DataFrame,
    by_season: pd.DataFrame,
    by_species: pd.DataFrame,
    by_species_season: pd.DataFrame,
    effort_support: pd.DataFrame,
    focus_species: list[str],
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(output_dir / "overall_summary.csv", index=False)
    by_season.to_csv(output_dir / "season_summary.csv", index=False)
    by_species.to_csv(output_dir / "species_replication_summary.csv", index=False)
    by_species_season.to_csv(output_dir / "species_season_summary.csv", index=False)
    effort_support.to_csv(output_dir / "effort_support_summary.csv", index=False)
    focus = by_species_season.loc[
        by_species_season["common_name"].isin(focus_species)
    ].copy()
    focus.to_csv(output_dir / "focus_species_season_summary.csv", index=False)
    (output_dir / "diagnostic_metadata.json").write_text(
        json.dumps(
            {
                "dataset_metadata": metadata,
                "focus_species": focus_species,
                "outputs": [
                    "overall_summary.csv",
                    "season_summary.csv",
                    "species_replication_summary.csv",
                    "species_season_summary.csv",
                    "effort_support_summary.csv",
                    "focus_species_season_summary.csv",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else dataset_dir / "diagnostics" / "replication_support"
    )
    locality_seasons = pd.read_parquet(dataset_dir / "locality_seasons.parquet")
    triplets = pd.read_parquet(dataset_dir / "locality_season_species.parquet")
    metadata = read_metadata(dataset_dir)

    overall = summarize_overall(locality_seasons, triplets)
    by_season = summarize_by_season(locality_seasons, triplets)
    by_species = summarize_by_species(triplets)
    by_species_season = summarize_species_season(
        triplets, args.min_positive_locality_seasons
    )
    effort_support = summarize_effort_support(locality_seasons)
    write_outputs(
        output_dir,
        overall,
        by_season,
        by_species,
        by_species_season,
        effort_support,
        args.focus_species,
        metadata,
    )

    print(f"Wrote locality-season replication diagnostics to {output_dir}")
    print()
    print("Overall:")
    print(overall.to_string(index=False))
    print()
    print("Season support:")
    print(by_season.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print("Focus species by season:")
    focus = by_species_season.loc[
        by_species_season["common_name"].isin(args.focus_species)
    ].copy()
    print(
        focus[
            [
                "common_name",
                "season_name",
                "positive_locality_seasons",
                "adequately_sampled_locality_seasons",
                "detections",
                "mean_detection_rate",
                "positive_support_ok",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
    )


if __name__ == "__main__":
    main()
