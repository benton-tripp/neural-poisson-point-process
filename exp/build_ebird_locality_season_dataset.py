"""
Build locality-season replication tables from processed complete-checklist eBird data.

The output is intended as the next bridge from checklist-level detection
prediction toward occupancy/detection modeling. It keeps repeated visits grouped
by locality and biologically meaningful season windows, then crosses eligible
locality-season groups with the top species so zero-detection species are
retained.

Run from the project root:

    python exp/build_ebird_locality_season_dataset.py --processed-dir data/ebird/processed_nc_2020_2023 --output-dir data/ebird/locality_season_top100 --top-species 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


DEFAULT_PROCESSED_DIR = "data/ebird/processed_nc_2020_2023"
DEFAULT_OUTPUT_DIR = "data/ebird/locality_season_top100"
DEFAULT_LOCALITY_TYPES = ["H", "P"]

CHECKLIST_COLUMNS = [
    "sampling_event_identifier",
    "locality_id",
    "locality",
    "locality_type",
    "county",
    "county_code",
    "observation_date",
    "year",
    "month",
    "day_of_year",
    "time_observations_started",
    "protocol_code",
    "protocol_name",
    "duration_minutes",
    "effort_distance_km",
    "number_observers",
    "observer_id",
    "canopy_median",
    "nc_usgs30m_match_tcc",
    "distance_to_waterbody_m",
    "distance_to_coastline_m",
]

SEASON_ORDER = [
    "winter",
    "spring_migration",
    "early_breeding",
    "late_breeding",
    "fall_migration",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build locality-season-species replication tables from processed eBird outputs."
    )
    parser.add_argument(
        "--processed-dir",
        default=DEFAULT_PROCESSED_DIR,
        help=f"Directory with checklists.geoparquet, detections.parquet, and species.csv. Defaults to {DEFAULT_PROCESSED_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--top-species",
        type=int,
        default=100,
        help="Number of most frequently detected species to include. Defaults to 100.",
    )
    parser.add_argument(
        "--season-scheme",
        choices=["biological-nc", "monthly"],
        default="biological-nc",
        help="Season grouping scheme. Defaults to biological-nc.",
    )
    parser.add_argument(
        "--include-locality-type",
        action="append",
        default=None,
        help=(
            "eBird locality type to include. Repeat as needed. Defaults to H and P. "
            "Use --include-all-locality-types to disable this filter."
        ),
    )
    parser.add_argument(
        "--include-all-locality-types",
        action="store_true",
        help="Include all locality types instead of the default hotspot/personal filter.",
    )
    parser.add_argument(
        "--min-checklists",
        type=int,
        default=3,
        help="Minimum checklists required for a locality-season to be crossed with species. Defaults to 3.",
    )
    parser.add_argument(
        "--min-dates",
        type=int,
        default=2,
        help="Minimum unique dates for the adequate_sampling flag. Defaults to 2.",
    )
    parser.add_argument(
        "--min-effort-bins",
        type=int,
        default=2,
        help="Minimum occupied duration bins for the adequate_sampling flag. Defaults to 2.",
    )
    parser.add_argument(
        "--max-checklists",
        type=int,
        default=None,
        help="Optional checklist row limit for smoke tests. Defaults to all rows.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {path}. Use --overwrite to replace files."
        )
    path.mkdir(parents=True, exist_ok=True)


def load_checklists(processed_dir: Path, max_checklists: int | None) -> gpd.GeoDataFrame:
    checklists = gpd.read_parquet(processed_dir / "checklists.geoparquet")
    if max_checklists is not None:
        checklists = checklists.head(max_checklists).copy()
    missing = [col for col in CHECKLIST_COLUMNS if col not in checklists.columns]
    if missing:
        raise ValueError(f"Missing checklist columns: {', '.join(missing)}")
    return checklists


def assign_biological_nc_season(frame: pd.DataFrame) -> pd.DataFrame:
    day = frame["day_of_year"].astype(int).to_numpy()
    year = frame["year"].astype(int).to_numpy()
    season = np.select(
        [
            (day >= 335) | (day <= 59),
            day <= 120,
            day <= 181,
            day <= 243,
        ],
        [
            "winter",
            "spring_migration",
            "early_breeding",
            "late_breeding",
        ],
        default="fall_migration",
    )
    season_year = np.where(day >= 335, year + 1, year)
    out = frame.copy()
    out["season_name"] = pd.Categorical(season, categories=SEASON_ORDER, ordered=True)
    out["season_year"] = season_year.astype(np.int16)
    out["season_key"] = out["season_year"].astype(str) + "_" + out["season_name"].astype(str)
    return out


def assign_monthly_season(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    month = out["month"].astype(int)
    out["season_name"] = "month_" + month.astype(str).str.zfill(2)
    out["season_year"] = out["year"].astype(np.int16)
    out["season_key"] = out["season_year"].astype(str) + "_" + out["season_name"]
    return out


def add_season_columns(frame: pd.DataFrame, scheme: str) -> pd.DataFrame:
    if scheme == "biological-nc":
        return assign_biological_nc_season(frame)
    if scheme == "monthly":
        return assign_monthly_season(frame)
    raise ValueError(f"Unknown season scheme: {scheme}")


def parse_start_hour(values: pd.Series) -> pd.Series:
    text = values.astype("string")
    parsed = pd.to_datetime(text, format="%H:%M:%S", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    return parsed.dt.hour + parsed.dt.minute / 60.0


def duration_bin_codes(values: pd.Series) -> pd.Series:
    bins = [-np.inf, 10, 30, 60, 120, np.inf]
    labels = ["1_10", "11_30", "31_60", "61_120", "121_plus"]
    return pd.cut(values.astype(float), bins=bins, labels=labels)


def prepare_checklists(args: argparse.Namespace) -> gpd.GeoDataFrame:
    checklists = load_checklists(Path(args.processed_dir), args.max_checklists)
    before = len(checklists)
    checklists = checklists.loc[checklists["locality_id"].notna()].copy()
    if not args.include_all_locality_types:
        locality_types = args.include_locality_type or DEFAULT_LOCALITY_TYPES
        checklists = checklists.loc[checklists["locality_type"].isin(locality_types)].copy()
    checklists = add_season_columns(checklists, args.season_scheme)
    checklists["start_hour"] = parse_start_hour(checklists["time_observations_started"])
    checklists["duration_bin"] = duration_bin_codes(checklists["duration_minutes"])
    checklists["is_stationary"] = checklists["protocol_code"].eq("P21")
    checklists["is_traveling"] = checklists["protocol_code"].eq("P22")
    checklists["x"] = checklists.geometry.x.astype(float)
    checklists["y"] = checklists.geometry.y.astype(float)
    checklists["observation_date_parsed"] = pd.to_datetime(
        checklists["observation_date"], errors="coerce"
    )
    print(f"Retained {len(checklists):,} of {before:,} checklists after locality filters.")
    return checklists


def summarize_locality_seasons(checklists: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    group_cols = ["locality_id", "season_year", "season_name", "season_key"]
    grouped = checklists.groupby(group_cols, observed=True, sort=True)
    summary = grouped.agg(
        n_checklists=("sampling_event_identifier", "nunique"),
        n_dates=("observation_date", "nunique"),
        first_date=("observation_date_parsed", "min"),
        last_date=("observation_date_parsed", "max"),
        locality=("locality", "first"),
        locality_type=("locality_type", "first"),
        county=("county", "first"),
        county_code=("county_code", "first"),
        x=("x", "median"),
        y=("y", "median"),
        unique_observers=("observer_id", "nunique"),
        duration_mean=("duration_minutes", "mean"),
        duration_median=("duration_minutes", "median"),
        duration_p90=("duration_minutes", lambda s: s.quantile(0.9)),
        effort_distance_mean=("effort_distance_km", "mean"),
        effort_distance_median=("effort_distance_km", "median"),
        effort_distance_p90=("effort_distance_km", lambda s: s.quantile(0.9)),
        number_observers_mean=("number_observers", "mean"),
        number_observers_median=("number_observers", "median"),
        start_hour_mean=("start_hour", "mean"),
        start_hour_median=("start_hour", "median"),
        stationary_rate=("is_stationary", "mean"),
        traveling_rate=("is_traveling", "mean"),
        duration_bin_count=("duration_bin", "nunique"),
        protocol_count=("protocol_code", "nunique"),
        canopy_median=("canopy_median", "median"),
        elevation_median=("nc_usgs30m_match_tcc", "median"),
        distance_to_waterbody_m_median=("distance_to_waterbody_m", "median"),
        distance_to_coastline_m_median=("distance_to_coastline_m", "median"),
    ).reset_index()

    summary.insert(0, "locality_season_id", np.arange(len(summary), dtype=np.int64))
    summary["observer_per_checklist"] = (
        summary["unique_observers"] / summary["n_checklists"].clip(lower=1)
    )
    summary["adequate_sampling"] = (
        (summary["n_checklists"] >= args.min_checklists)
        & (summary["n_dates"] >= args.min_dates)
        & (summary["duration_bin_count"] >= args.min_effort_bins)
    )
    summary["eligible_for_species_table"] = summary["n_checklists"] >= args.min_checklists
    return summary


def load_species(processed_dir: Path, top_species: int) -> pd.DataFrame:
    species = pd.read_csv(processed_dir / "species.csv").head(top_species).copy()
    species.insert(0, "species_index", np.arange(len(species), dtype=np.int16))
    return species


def load_detection_counts(
    processed_dir: Path,
    checklists: pd.DataFrame,
    species: pd.DataFrame,
    locality_seasons: pd.DataFrame,
) -> pd.DataFrame:
    event_to_group = checklists[["sampling_event_identifier", "locality_id", "season_year", "season_name"]].merge(
        locality_seasons[
            ["locality_season_id", "locality_id", "season_year", "season_name"]
        ],
        on=["locality_id", "season_year", "season_name"],
        how="left",
        validate="many_to_one",
    )[["sampling_event_identifier", "locality_season_id"]]
    species_keys = set(species["species_key"])
    detections = pd.read_parquet(
        processed_dir / "detections.parquet",
        columns=["sampling_event_identifier", "species_key"],
    )
    detections = detections.loc[detections["species_key"].isin(species_keys)]
    detections = detections.merge(
        event_to_group,
        on="sampling_event_identifier",
        how="inner",
        validate="many_to_one",
    )
    counts = (
        detections.drop_duplicates(
            ["locality_season_id", "sampling_event_identifier", "species_key"]
        )
        .groupby(["locality_season_id", "species_key"], observed=True)
        .size()
        .rename("n_detections")
        .reset_index()
    )
    return counts


def build_species_triplets(
    locality_seasons: pd.DataFrame,
    detection_counts: pd.DataFrame,
    species: pd.DataFrame,
    min_checklists: int,
) -> pd.DataFrame:
    eligible = locality_seasons.loc[
        locality_seasons["n_checklists"] >= min_checklists
    ].copy()
    locality_ids = eligible["locality_season_id"].to_numpy(dtype=np.int64)
    species_indices = species["species_index"].to_numpy(dtype=np.int16)
    species_keys = species["species_key"].to_numpy()

    triplets = pd.DataFrame(
        {
            "locality_season_id": np.repeat(locality_ids, len(species)),
            "species_index": np.tile(species_indices, len(eligible)),
            "species_key": np.tile(species_keys, len(eligible)),
        }
    )
    triplets = triplets.merge(
        detection_counts,
        on=["locality_season_id", "species_key"],
        how="left",
    )
    triplets["n_detections"] = triplets["n_detections"].fillna(0).astype(np.int16)

    season_cols = [
        "locality_season_id",
        "locality_id",
        "season_year",
        "season_name",
        "season_key",
        "n_checklists",
        "n_dates",
        "unique_observers",
        "duration_mean",
        "duration_median",
        "duration_p90",
        "effort_distance_mean",
        "effort_distance_median",
        "effort_distance_p90",
        "number_observers_mean",
        "stationary_rate",
        "traveling_rate",
        "duration_bin_count",
        "protocol_count",
        "adequate_sampling",
        "x",
        "y",
        "canopy_median",
        "elevation_median",
        "distance_to_waterbody_m_median",
        "distance_to_coastline_m_median",
    ]
    triplets = triplets.merge(
        locality_seasons[season_cols],
        on="locality_season_id",
        how="left",
        validate="many_to_one",
    )
    triplets = triplets.merge(
        species[["species_index", "common_name", "scientific_name"]],
        on="species_index",
        how="left",
        validate="many_to_one",
    )
    triplets["n_non_detections"] = (
        triplets["n_checklists"] - triplets["n_detections"]
    ).astype(np.int16)
    triplets["naive_detection_rate"] = (
        triplets["n_detections"] / triplets["n_checklists"].clip(lower=1)
    ).astype(np.float32)
    return triplets


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    checklists: pd.DataFrame,
    locality_seasons: pd.DataFrame,
    triplets: pd.DataFrame,
    species: pd.DataFrame,
) -> None:
    metadata = {
        "processed_dir": str(Path(args.processed_dir)),
        "top_species": int(args.top_species),
        "season_scheme": args.season_scheme,
        "included_locality_types": "all"
        if args.include_all_locality_types
        else (args.include_locality_type or DEFAULT_LOCALITY_TYPES),
        "min_checklists": int(args.min_checklists),
        "min_dates": int(args.min_dates),
        "min_effort_bins": int(args.min_effort_bins),
        "checklists_retained": int(len(checklists)),
        "locality_seasons": int(len(locality_seasons)),
        "eligible_locality_seasons": int(
            locality_seasons["eligible_for_species_table"].sum()
        ),
        "adequately_sampled_locality_seasons": int(
            locality_seasons["adequate_sampling"].sum()
        ),
        "species": int(len(species)),
        "locality_season_species_rows": int(len(triplets)),
        "species_triplets_with_detection": int((triplets["n_detections"] > 0).sum()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    pd.DataFrame([metadata]).to_csv(output_dir / "summary.csv", index=False)


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    ensure_output_dir(output_dir, args.overwrite)

    checklists = prepare_checklists(args)
    species = load_species(processed_dir, args.top_species)
    locality_seasons = summarize_locality_seasons(checklists, args)
    detection_counts = load_detection_counts(
        processed_dir, checklists, species, locality_seasons
    )
    triplets = build_species_triplets(
        locality_seasons, detection_counts, species, args.min_checklists
    )

    locality_seasons.to_parquet(output_dir / "locality_seasons.parquet", index=False)
    triplets.to_parquet(output_dir / "locality_season_species.parquet", index=False)
    species.to_csv(output_dir / "species.csv", index=False)
    write_metadata(output_dir, args, checklists, locality_seasons, triplets, species)

    print("Locality-season replication dataset written:")
    print(f"  checklists_retained: {len(checklists):,}")
    print(f"  locality_seasons: {len(locality_seasons):,}")
    print(
        "  eligible_locality_seasons: "
        f"{int(locality_seasons['eligible_for_species_table'].sum()):,}"
    )
    print(
        "  adequately_sampled_locality_seasons: "
        f"{int(locality_seasons['adequate_sampling'].sum()):,}"
    )
    print(f"  species: {len(species):,}")
    print(f"  locality_season_species_rows: {len(triplets):,}")
    print(f"  output_dir: {output_dir}")


if __name__ == "__main__":
    main()
