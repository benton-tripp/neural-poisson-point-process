"""
Preprocess an eBird Basic Dataset extract for joint effort-aware SDM work.

The script creates a checklist/location-time table and a species-detection edge
table from the all-species EBD observation file plus the Sampling Event Data
file. The checklist table can be written as GeoParquet and optionally enriched
with raster covariates.

Examples:

    python scripts/data/preprocess-ebird-bulk.py --ebd-dir data/ebird/ebd_US-NC_202001_202312_smp_relApr-2026 --output-dir data/ebird/processed_nc_2020_2023 --raster data/nc_covariate_stack.tif

    python scripts/data/preprocess-ebird-bulk.py --output-dir data/ebird/processed_nc_2020_2023 --boundary data/boundaries/nc_state_boundary.gpkg --update boundary effort-distance canopy linked-tables

    python scripts/data/preprocess-ebird-bulk.py --ebd-dir data/ebird/ebd_US-NC_202001_202312_smp_relApr-2026 --output-dir data/ebird/processed_nc_2020_2023 --protocol-code P21 --protocol-code P22 --protocol-code P23 --category species --category issf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from shapely.ops import unary_union

try:
    import pyarrow  # noqa: F401
except ImportError:  # pragma: no cover - import guard for CLI users
    pyarrow = None


SOURCE_CRS = "EPSG:4326"
DEFAULT_CHUNKSIZE = 250_000
DEFAULT_PROTOCOL_CODES = ("P21", "P22")
DEFAULT_CATEGORIES = ("species",)
DEFAULT_SAMPLING_FILE_GLOB = "*_sampling.txt"
DEFAULT_EBD_FILE_GLOB = "ebd_*.txt"
DEFAULT_OUTPUT_CRS = "EPSG:5070"
DEFAULT_STATIONARY_DISTANCE = "null"
DEFAULT_TCC_MAX_CANOPY = 100.0
UPDATE_STEPS = ("boundary", "effort-distance", "canopy", "linked-tables", "species")

SAMPLING_COLUMNS = [
    "LAST EDITED DATE",
    "COUNTRY",
    "COUNTRY CODE",
    "STATE",
    "STATE CODE",
    "COUNTY",
    "COUNTY CODE",
    "IBA CODE",
    "BCR CODE",
    "USFWS CODE",
    "ATLAS BLOCK",
    "LOCALITY",
    "LOCALITY ID",
    "LOCALITY TYPE",
    "LATITUDE",
    "LONGITUDE",
    "OBSERVATION DATE",
    "TIME OBSERVATIONS STARTED",
    "OBSERVER ID",
    "OBSERVER ORCID ID",
    "SAMPLING EVENT IDENTIFIER",
    "OBSERVATION TYPE",
    "PROTOCOL NAME",
    "PROTOCOL CODE",
    "PROJECT NAMES",
    "PROJECT IDENTIFIERS",
    "DURATION MINUTES",
    "EFFORT DISTANCE KM",
    "EFFORT AREA HA",
    "NUMBER OBSERVERS",
    "ALL SPECIES REPORTED",
    "GROUP IDENTIFIER",
]

DETECTION_COLUMNS = [
    "TAXONOMIC ORDER",
    "CATEGORY",
    "TAXON CONCEPT ID",
    "COMMON NAME",
    "SCIENTIFIC NAME",
    "OBSERVATION COUNT",
    "SAMPLING EVENT IDENTIFIER",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare eBird bulk data as checklist nodes and species-detection edges."
    )
    parser.add_argument(
        "--ebd-dir",
        help="Directory containing the eBird EBD .txt file and *_sampling.txt file.",
    )
    parser.add_argument(
        "--ebd-file",
        help="Optional explicit EBD observation file. Defaults to the ebd_*.txt file in --ebd-dir.",
    )
    parser.add_argument(
        "--sampling-file",
        help="Optional explicit Sampling Event Data file. Defaults to *_sampling.txt in --ebd-dir.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for processed outputs.",
    )
    parser.add_argument(
        "--output-crs",
        default=DEFAULT_OUTPUT_CRS,
        help=f"CRS for checklist GeoParquet output. Defaults to {DEFAULT_OUTPUT_CRS}.",
    )
    parser.add_argument(
        "--raster",
        help="Optional raster stack to sample onto retained checklists.",
    )
    parser.add_argument(
        "--boundary",
        help=(
            "Optional boundary vector path readable by GeoPandas. Full runs keep "
            "checklists intersecting the dissolved boundary. Update runs use this "
            "with --update boundary."
        ),
    )
    parser.add_argument(
        "--protocol-code",
        action="append",
        dest="protocol_codes",
        help=(
            "Protocol code to retain. Can be repeated. "
            f"Defaults to {', '.join(DEFAULT_PROTOCOL_CODES)}."
        ),
    )
    parser.add_argument(
        "--category",
        action="append",
        dest="categories",
        help=(
            "Taxonomic category to retain for detection edges. Can be repeated. "
            f"Defaults to {', '.join(DEFAULT_CATEGORIES)}."
        ),
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Keep incomplete checklists instead of requiring ALL SPECIES REPORTED = 1.",
    )
    parser.add_argument(
        "--no-deduplicate-groups",
        action="store_true",
        help="Do not collapse shared checklists with the same GROUP IDENTIFIER.",
    )
    parser.add_argument(
        "--min-duration-minutes",
        type=float,
        default=0.0,
        help="Minimum checklist duration. Defaults to 0.",
    )
    parser.add_argument(
        "--max-duration-minutes",
        type=float,
        default=300.0,
        help="Maximum checklist duration. Defaults to 300.",
    )
    parser.add_argument(
        "--max-travel-distance-km",
        type=float,
        default=10.0,
        help="Maximum effort distance for traveling checklists. Defaults to 10.",
    )
    parser.add_argument(
        "--stationary-distance",
        choices=("keep", "null", "zero"),
        default=DEFAULT_STATIONARY_DISTANCE,
        help=(
            "How to handle EFFORT DISTANCE KM on stationary checklists. "
            "Defaults to null because stationary counts have no modeled travel distance."
        ),
    )
    parser.add_argument(
        "--max-observers",
        type=float,
        default=20.0,
        help="Maximum number of observers. Defaults to 20.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help=f"Rows per chunk while streaming large text files. Defaults to {DEFAULT_CHUNKSIZE}.",
    )
    parser.add_argument(
        "--write-detections-geo",
        action="store_true",
        help="Also write detections as GeoParquet with checklist geometry repeated.",
    )
    parser.add_argument(
        "--tcc-max-canopy",
        type=float,
        default=DEFAULT_TCC_MAX_CANOPY,
        help=(
            "Maximum valid TCC percent canopy value. TCC values above this are "
            f"set to missing before canopy_median is computed. Defaults to {DEFAULT_TCC_MAX_CANOPY:g}."
        ),
    )
    parser.add_argument(
        "--drop-missing-raster-covariates",
        choices=("none", "any", "all"),
        default="none",
        help=(
            "Drop checklists after raster sampling based on sampled raster "
            "covariates. Use 'any' to make the raster stack's valid-data "
            "footprint the effective checklist mask. Defaults to none."
        ),
    )
    parser.add_argument(
        "--update",
        nargs="+",
        choices=UPDATE_STEPS,
        help=(
            "Update existing processed outputs without re-streaming the raw EBD. "
            "Steps run in dependency order. Choices: boundary, effort-distance, "
            "canopy, linked-tables, species."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def snake_case(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def find_one_file(directory: Path, pattern: str, exclude_sampling: bool = False) -> Path:
    candidates = sorted(directory.glob(pattern))
    if exclude_sampling:
        candidates = [path for path in candidates if "_sampling" not in path.name]
    if len(candidates) != 1:
        names = ", ".join(path.name for path in candidates[:10])
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern!r} in {directory}; found {len(candidates)}: {names}"
        )
    return candidates[0]


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.ebd_dir is None:
        raise ValueError("--ebd-dir is required for a full preprocessing run.")

    ebd_dir = Path(args.ebd_dir)
    if not ebd_dir.exists():
        raise FileNotFoundError(f"EBD directory does not exist: {ebd_dir}")

    ebd_file = Path(args.ebd_file) if args.ebd_file else find_one_file(
        ebd_dir, DEFAULT_EBD_FILE_GLOB, exclude_sampling=True
    )
    sampling_file = Path(args.sampling_file) if args.sampling_file else find_one_file(
        ebd_dir, DEFAULT_SAMPLING_FILE_GLOB
    )
    output_dir = Path(args.output_dir)
    return ebd_file, sampling_file, output_dir


def ensure_outputs_can_be_written(output_dir: Path, overwrite: bool) -> None:
    outputs = [
        output_dir / "checklists.geoparquet",
        output_dir / "detections.parquet",
        output_dir / "species.csv",
        output_dir / "preprocessing_summary.csv",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output file(s) already exist. Use --overwrite to replace: {names}")
    output_dir.mkdir(parents=True, exist_ok=True)


def ensure_update_outputs_exist(output_dir: Path) -> None:
    required = [output_dir / "checklists.geoparquet"]
    missing = [path for path in required if not path.exists()]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing processed output(s) for --update: {names}")


def read_tab_chunks(path: Path, usecols: list[str], chunksize: int):
    return pd.read_csv(
        path,
        sep="\t",
        usecols=usecols,
        dtype="string",
        chunksize=chunksize,
        low_memory=False,
    )


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")


def filter_sampling_chunk(
    chunk: pd.DataFrame,
    protocol_codes: set[str],
    include_incomplete: bool,
    min_duration_minutes: float,
    max_duration_minutes: float,
    max_travel_distance_km: float,
    max_observers: float,
) -> pd.DataFrame:
    chunk = chunk.rename(columns={column: snake_case(column) for column in chunk.columns})
    coerce_numeric(
        chunk,
        [
            "latitude",
            "longitude",
            "duration_minutes",
            "effort_distance_km",
            "effort_area_ha",
            "number_observers",
            "all_species_reported",
        ],
    )

    mask = chunk["latitude"].between(-90, 90) & chunk["longitude"].between(-180, 180)
    if not include_incomplete:
        mask &= chunk["all_species_reported"].eq(1)
    if protocol_codes:
        mask &= chunk["protocol_code"].isin(protocol_codes)

    duration = chunk["duration_minutes"]
    mask &= duration.gt(min_duration_minutes) & duration.le(max_duration_minutes)

    observers = chunk["number_observers"]
    mask &= observers.isna() | observers.le(max_observers)

    is_traveling = chunk["protocol_code"].eq("P22")
    distance = chunk["effort_distance_km"]
    mask &= ~is_traveling | distance.isna() | distance.le(max_travel_distance_km)

    return chunk.loc[mask].copy()


def apply_stationary_distance_policy(
    checklists: gpd.GeoDataFrame | pd.DataFrame,
    policy: str,
) -> gpd.GeoDataFrame | pd.DataFrame:
    if policy == "keep" or "effort_distance_km" not in checklists.columns:
        return checklists

    output = checklists.copy()
    stationary = output["protocol_code"].eq("P21")
    if policy == "null":
        output.loc[stationary, "effort_distance_km"] = pd.NA
    elif policy == "zero":
        output.loc[stationary, "effort_distance_km"] = 0
    else:
        raise ValueError(f"Unsupported stationary distance policy: {policy}")
    return output


def add_temporal_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date = pd.to_datetime(df["observation_date"], errors="coerce")
    df["observation_date"] = date.dt.date.astype("string")
    df["year"] = date.dt.year.astype("Int16")
    df["month"] = date.dt.month.astype("Int8")
    df["day_of_year"] = date.dt.dayofyear.astype("Int16")
    df["day_of_week"] = date.dt.dayofweek.astype("Int8")
    return df


def deduplicate_group_checklists(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_dedupe_key"] = df["group_identifier"].fillna(df["sampling_event_identifier"])
    df["_dedupe_key"] = df["_dedupe_key"].mask(
        df["_dedupe_key"].eq(""), df["sampling_event_identifier"]
    )
    df = df.sort_values(["_dedupe_key", "sampling_event_identifier"])
    df = df.drop_duplicates("_dedupe_key", keep="first")
    return df.drop(columns="_dedupe_key")


def load_checklists(
    sampling_file: Path,
    protocol_codes: set[str],
    include_incomplete: bool,
    deduplicate_groups: bool,
    min_duration_minutes: float,
    max_duration_minutes: float,
    max_travel_distance_km: float,
    max_observers: float,
    chunksize: int,
) -> gpd.GeoDataFrame:
    frames = []
    rows_seen = 0
    for chunk in read_tab_chunks(sampling_file, SAMPLING_COLUMNS, chunksize):
        rows_seen += len(chunk)
        filtered = filter_sampling_chunk(
            chunk,
            protocol_codes=protocol_codes,
            include_incomplete=include_incomplete,
            min_duration_minutes=min_duration_minutes,
            max_duration_minutes=max_duration_minutes,
            max_travel_distance_km=max_travel_distance_km,
            max_observers=max_observers,
        )
        if not filtered.empty:
            frames.append(filtered)
        print(f"Sampling rows scanned: {rows_seen:,}; retained so far: {sum(len(f) for f in frames):,}")

    if not frames:
        raise ValueError("No checklists remained after filtering.")

    checklists = pd.concat(frames, ignore_index=True)
    before_dedupe = len(checklists)
    if deduplicate_groups:
        checklists = deduplicate_group_checklists(checklists)
        print(f"Deduplicated shared checklist groups: {before_dedupe:,} -> {len(checklists):,}")

    checklists = add_temporal_columns(checklists)
    checklists = gpd.GeoDataFrame(
        checklists,
        geometry=gpd.points_from_xy(checklists["longitude"], checklists["latitude"]),
        crs=SOURCE_CRS,
    )
    return checklists


def load_boundary_geometry(boundary_path: Path, target_crs: object) -> object:
    if not boundary_path.exists():
        raise FileNotFoundError(f"Boundary file does not exist: {boundary_path}")

    boundary = gpd.read_file(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary file has no features: {boundary_path}")
    if boundary.crs is None:
        raise ValueError(f"Boundary file has no CRS: {boundary_path}")

    boundary = boundary[boundary.geometry.notna()].copy()
    boundary = boundary[~boundary.geometry.is_empty].copy()
    if boundary.empty:
        raise ValueError(f"Boundary file has no valid geometries: {boundary_path}")

    boundary = boundary.to_crs(target_crs)
    if hasattr(boundary.geometry, "union_all"):
        return boundary.geometry.union_all()
    return unary_union(list(boundary.geometry))


def filter_checklists_to_boundary(
    checklists: gpd.GeoDataFrame,
    boundary_path: Path,
) -> gpd.GeoDataFrame:
    boundary_geom = load_boundary_geometry(boundary_path, checklists.crs)
    before = len(checklists)
    filtered = checklists.loc[checklists.geometry.intersects(boundary_geom)].copy()
    print(f"Boundary filter kept {len(filtered):,} of {before:,} checklists.")
    if filtered.empty:
        raise ValueError("Boundary filter removed all checklists.")
    return filtered


def raster_band_names(src: rasterio.DatasetReader) -> list[str]:
    names = []
    seen: dict[str, int] = {}
    for index, description in enumerate(src.descriptions, start=1):
        name = snake_case(description) if description else f"band_{index}"
        count = seen.get(name, 0)
        seen[name] = count + 1
        names.append(name if count == 0 else f"{name}_{count + 1}")
    return names


def sample_raster_covariates(
    checklists: gpd.GeoDataFrame,
    raster_path: Path,
    tcc_max_canopy: float,
) -> gpd.GeoDataFrame:
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file does not exist: {raster_path}")

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")
        band_names = raster_band_names(src)
        raster_points = checklists.to_crs(src.crs)
        coordinates = [(geom.x, geom.y) for geom in raster_points.geometry]
        sampled = np.asarray(list(src.sample(coordinates, masked=True)), dtype=np.float64)
        if np.ma.isMaskedArray(sampled):
            sampled = sampled.filled(np.nan)
        for band_index, nodata in enumerate(src.nodatavals):
            if nodata is not None:
                sampled[:, band_index] = np.where(sampled[:, band_index] == nodata, np.nan, sampled[:, band_index])

    output = checklists.copy()
    for column, values in zip(band_names, sampled.T):
        output[column] = values

    return clean_tcc_columns(output, max_canopy=tcc_max_canopy)


def clean_tcc_columns(
    checklists: gpd.GeoDataFrame,
    max_canopy: float,
) -> gpd.GeoDataFrame:
    output = checklists.copy()
    canopy_columns = [column for column in output.columns if column.startswith("tcc_")]
    if canopy_columns:
        for column in canopy_columns:
            values = pd.to_numeric(output[column], errors="coerce")
            output[column] = values.mask(values > max_canopy)
        output["canopy_median"] = output[canopy_columns].median(axis=1, skipna=True)
    return output


def drop_missing_raster_covariates(
    checklists: gpd.GeoDataFrame,
    raster_columns: list[str],
    mode: str,
) -> gpd.GeoDataFrame:
    if mode == "none" or not raster_columns:
        return checklists
    if mode == "any":
        missing = checklists[raster_columns].isna().any(axis=1)
    elif mode == "all":
        missing = checklists[raster_columns].isna().all(axis=1)
    else:
        raise ValueError(f"Unsupported missing raster covariate drop mode: {mode}")

    before = len(checklists)
    filtered = checklists.loc[~missing].copy()
    print(
        f"Raster covariate mask kept {len(filtered):,} of {before:,} checklists "
        f"using drop mode {mode!r}."
    )
    if filtered.empty:
        raise ValueError("Raster covariate mask removed all checklists.")
    return filtered


def parse_observation_count(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    present_code = series.astype("string").str.upper().eq("X")
    return numeric.mask(present_code, 1)


def load_detection_edges(
    ebd_file: Path,
    retained_event_ids: set[str],
    categories: set[str],
    chunksize: int,
) -> pd.DataFrame:
    frames = []
    rows_seen = 0
    for chunk in read_tab_chunks(ebd_file, DETECTION_COLUMNS, chunksize):
        rows_seen += len(chunk)
        chunk = chunk.rename(columns={column: snake_case(column) for column in chunk.columns})
        mask = chunk["sampling_event_identifier"].isin(retained_event_ids)
        if categories:
            mask &= chunk["category"].isin(categories)
        filtered = chunk.loc[mask].copy()
        if not filtered.empty:
            filtered["taxonomic_order"] = pd.to_numeric(filtered["taxonomic_order"], errors="coerce")
            filtered["observation_count_numeric"] = parse_observation_count(filtered["observation_count"])
            filtered["species_key"] = filtered["taxon_concept_id"].fillna(filtered["scientific_name"])
            frames.append(filtered)
        print(f"EBD rows scanned: {rows_seen:,}; detection edges retained so far: {sum(len(f) for f in frames):,}")

    if not frames:
        raise ValueError("No detection edges remained after filtering.")
    return pd.concat(frames, ignore_index=True)


def build_species_table(detections: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        detections.groupby(["species_key", "taxon_concept_id", "common_name", "scientific_name", "category"], dropna=False)
        .agg(
            detection_edges=("sampling_event_identifier", "size"),
            checklists_detected=("sampling_event_identifier", "nunique"),
            taxonomic_order=("taxonomic_order", "min"),
        )
        .reset_index()
        .sort_values(["checklists_detected", "detection_edges"], ascending=False)
    )
    return grouped


def read_existing_outputs(output_dir: Path) -> tuple[gpd.GeoDataFrame, pd.DataFrame | None]:
    checklists_path = output_dir / "checklists.geoparquet"
    detections_path = output_dir / "detections.parquet"
    checklists = gpd.read_parquet(checklists_path)
    detections = pd.read_parquet(detections_path) if detections_path.exists() else None
    return checklists, detections


def update_summary(output_dir: Path, checklists: gpd.GeoDataFrame, detections: pd.DataFrame | None) -> None:
    metrics = [
        {"metric": "checklists", "value": len(checklists)},
        {"metric": "output_crs", "value": str(checklists.crs)},
    ]
    if detections is not None:
        species = build_species_table(detections)
        metrics.extend(
            [
                {"metric": "detection_edges", "value": len(detections)},
                {"metric": "species", "value": len(species)},
            ]
        )
    pd.DataFrame(metrics).to_csv(output_dir / "preprocessing_summary.csv", index=False)


def run_update(args: argparse.Namespace, output_dir: Path) -> None:
    if pyarrow is None:
        raise RuntimeError(
            "pyarrow is required to update Parquet/GeoParquet outputs. "
            "Install project requirements again, or run: pip install pyarrow"
        )

    ensure_update_outputs_exist(output_dir)
    checklists, detections = read_existing_outputs(output_dir)
    steps = list(dict.fromkeys(args.update))

    if "boundary" in steps:
        if not args.boundary:
            raise ValueError("--boundary is required for --update boundary.")
        checklists = filter_checklists_to_boundary(checklists, Path(args.boundary))

    if "effort-distance" in steps:
        checklists = apply_stationary_distance_policy(checklists, args.stationary_distance)
        print(f"Applied stationary distance policy: {args.stationary_distance}")

    if "canopy" in steps:
        checklists = clean_tcc_columns(checklists, max_canopy=args.tcc_max_canopy)
        print(f"Masked TCC values above {args.tcc_max_canopy:g} and recomputed canopy_median.")

    if "linked-tables" in steps:
        if detections is None:
            raise FileNotFoundError(f"Missing detections table: {output_dir / 'detections.parquet'}")
        retained_event_ids = set(checklists["sampling_event_identifier"].dropna().astype(str))
        before = len(detections)
        detections = detections.loc[
            detections["sampling_event_identifier"].astype(str).isin(retained_event_ids)
        ].copy()
        print(f"Linked-table update kept {len(detections):,} of {before:,} detection edges.")
        species = build_species_table(detections)
        detections.to_parquet(output_dir / "detections.parquet", index=False)
        species.to_csv(output_dir / "species.csv", index=False)
        print(f"Rebuilt species table with {len(species):,} rows.")

    if "species" in steps and "linked-tables" not in steps:
        if detections is None:
            raise FileNotFoundError(f"Missing detections table: {output_dir / 'detections.parquet'}")
        species = build_species_table(detections)
        species.to_csv(output_dir / "species.csv", index=False)
        print(f"Rebuilt species table with {len(species):,} rows.")

    checklists.to_parquet(output_dir / "checklists.geoparquet", index=False)
    update_summary(output_dir, checklists, detections)
    print(f"Updated {output_dir / 'checklists.geoparquet'}")


def write_outputs(
    checklists: gpd.GeoDataFrame,
    detections: pd.DataFrame,
    species: pd.DataFrame,
    output_dir: Path,
    output_crs: str,
    write_detections_geo: bool,
) -> None:
    if pyarrow is None:
        raise RuntimeError(
            "pyarrow is required to write Parquet/GeoParquet outputs. "
            "Install project requirements again, or run: pip install pyarrow"
        )

    projected_checklists = checklists.to_crs(output_crs)
    checklists_path = output_dir / "checklists.geoparquet"
    detections_path = output_dir / "detections.parquet"
    species_path = output_dir / "species.csv"

    projected_checklists.to_parquet(checklists_path, index=False)
    detections.to_parquet(detections_path, index=False)
    species.to_csv(species_path, index=False)

    if write_detections_geo:
        geometry_columns = ["sampling_event_identifier", "geometry"]
        detections_geo = detections.merge(
            projected_checklists[geometry_columns],
            on="sampling_event_identifier",
            how="left",
            validate="many_to_one",
        )
        detections_geo = gpd.GeoDataFrame(detections_geo, geometry="geometry", crs=projected_checklists.crs)
        detections_geo.to_parquet(output_dir / "detections.geoparquet", index=False)

    summary = pd.DataFrame(
        [
            {"metric": "checklists", "value": len(checklists)},
            {"metric": "detection_edges", "value": len(detections)},
            {"metric": "species", "value": len(species)},
            {"metric": "output_crs", "value": str(projected_checklists.crs)},
        ]
    )
    summary.to_csv(output_dir / "preprocessing_summary.csv", index=False)
    print(f"Wrote {len(checklists):,} checklists to {checklists_path}")
    print(f"Wrote {len(detections):,} detection edges to {detections_path}")
    print(f"Wrote {len(species):,} species rows to {species_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.update:
        run_update(args, output_dir)
        return

    ebd_file, sampling_file, output_dir = resolve_inputs(args)
    ensure_outputs_can_be_written(output_dir, args.overwrite)

    protocol_codes = set(args.protocol_codes or DEFAULT_PROTOCOL_CODES)
    categories = set(args.categories or DEFAULT_CATEGORIES)
    print(f"EBD file: {ebd_file}")
    print(f"Sampling file: {sampling_file}")
    print(f"Retained protocol codes: {', '.join(sorted(protocol_codes))}")
    print(f"Retained categories: {', '.join(sorted(categories))}")

    checklists = load_checklists(
        sampling_file=sampling_file,
        protocol_codes=protocol_codes,
        include_incomplete=args.include_incomplete,
        deduplicate_groups=not args.no_deduplicate_groups,
        min_duration_minutes=args.min_duration_minutes,
        max_duration_minutes=args.max_duration_minutes,
        max_travel_distance_km=args.max_travel_distance_km,
        max_observers=args.max_observers,
        chunksize=args.chunksize,
    )
    checklists = apply_stationary_distance_policy(checklists, args.stationary_distance)
    if args.boundary:
        checklists = filter_checklists_to_boundary(checklists, Path(args.boundary))
    if args.raster:
        print(f"Sampling raster covariates from {args.raster}")
        checklists = sample_raster_covariates(
            checklists,
            Path(args.raster),
            tcc_max_canopy=args.tcc_max_canopy,
        )
        raster_columns = [
            column
            for column in checklists.columns
            if column.startswith("tcc_")
            or column in {
                "canopy_median",
                "nc_usgs30m_match_tcc",
                "distance_to_waterbody_m",
                "distance_to_coastline_m",
            }
        ]
        checklists = drop_missing_raster_covariates(
            checklists,
            raster_columns=raster_columns,
            mode=args.drop_missing_raster_covariates,
        )

    retained_event_ids = set(checklists["sampling_event_identifier"].dropna().astype(str))
    detections = load_detection_edges(
        ebd_file=ebd_file,
        retained_event_ids=retained_event_ids,
        categories=categories,
        chunksize=args.chunksize,
    )
    species = build_species_table(detections)
    write_outputs(
        checklists=checklists,
        detections=detections,
        species=species,
        output_dir=output_dir,
        output_crs=args.output_crs,
        write_detections_geo=args.write_detections_geo,
    )


if __name__ == "__main__":
    main()
