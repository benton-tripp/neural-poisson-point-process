"""
Build graph-ready node and edge tables from processed eBird bulk outputs.

Run from the project root:

    python exp/build_ebird_graph_dataset.py --processed-dir data/ebird/processed_nc_2020_2023 --output-dir data/ebird/graph_top100_spatial --top-species 100 --split spatial-stratified --spatial-blocks-per-dim 8 --test-fraction 0.2 --negative-ratio 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ebird_joint_tabular_baseline import (
    DEFAULT_PROCESSED_DIR,
    ECOLOGY_COLUMNS,
    SEED,
    build_features,
    build_labels,
    load_inputs,
    make_split,
)


DEFAULT_OUTPUT_DIR = "data/ebird/graph_top100_spatial"
RAW_CHECKLIST_COLUMNS = [
    "sampling_event_identifier",
    "observation_date",
    "year",
    "month",
    "day_of_year",
    "day_of_week",
    "protocol_code",
    "protocol_name",
    "duration_minutes",
    "effort_distance_km",
    "number_observers",
    "observer_id",
    "locality_id",
    "locality",
    "county",
    "county_code",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build graph-ready eBird checklist/species nodes and edge tables."
    )
    parser.add_argument(
        "--processed-dir",
        default=DEFAULT_PROCESSED_DIR,
        help=f"Directory with checklists.geoparquet, detections.parquet, and species.csv. Defaults to {DEFAULT_PROCESSED_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for graph dataset files. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--top-species",
        type=int,
        default=100,
        help="Number of most frequently detected species to include. Defaults to 100.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["effort", "ecology", "both"],
        default="both",
        help="Checklist feature set to export. Defaults to both.",
    )
    parser.add_argument(
        "--split",
        choices=["temporal", "spatial-stratified"],
        default="spatial-stratified",
        help="Train/test split strategy. Defaults to spatial-stratified.",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2023,
        help="Year held out for temporal split. Defaults to 2023.",
    )
    parser.add_argument(
        "--spatial-blocks-per-dim",
        type=int,
        default=8,
        help="Grid blocks per x/y dimension for spatial-stratified split. Defaults to 8.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Approximate checklist fraction held out for spatial-stratified split. Defaults to 0.2.",
    )
    parser.add_argument(
        "--stratify-species-count",
        type=int,
        default=20,
        help="Number of common species included in split balancing. Defaults to 20.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=SEED,
        help="Random seed used to break ties in spatial split selection. Defaults to 19.",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=5.0,
        help="Sample this many negative edges per positive edge on each checklist. Defaults to 5.",
    )
    parser.add_argument(
        "--negative-seed",
        type=int,
        default=SEED,
        help="Random seed for negative edge sampling. Defaults to 19.",
    )
    parser.add_argument(
        "--max-checklists",
        type=int,
        default=None,
        help="Optional row limit for smoke tests. Defaults to all checklists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty output directory.",
    )
    return parser.parse_args()


def standardize_features(
    features: pd.DataFrame,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    train = features.loc[train_mask].to_numpy(dtype=np.float32)
    values = features.to_numpy(dtype=np.float32)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    standardized = (values - mean) / std
    stats = {
        "feature_mean": mean.ravel().astype(float).tolist(),
        "feature_std": std.ravel().astype(float).tolist(),
    }
    return standardized.astype(np.float32), stats


def build_checklist_nodes(
    checklists: gpd.GeoDataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> pd.DataFrame:
    columns = [col for col in RAW_CHECKLIST_COLUMNS if col in checklists.columns]
    for col in ECOLOGY_COLUMNS:
        if col in checklists.columns and col not in columns:
            columns.append(col)

    nodes = checklists[columns].copy()
    nodes.insert(0, "checklist_index", np.arange(len(checklists), dtype=np.int32))
    nodes["feature_index"] = nodes["checklist_index"].astype(np.int32)
    nodes["x"] = checklists.geometry.x.astype(float)
    nodes["y"] = checklists.geometry.y.astype(float)
    nodes["train_mask"] = train_mask.astype(bool)
    nodes["test_mask"] = test_mask.astype(bool)
    return nodes


def build_species_nodes(species: pd.DataFrame) -> pd.DataFrame:
    nodes = species.copy()
    nodes.insert(0, "species_index", np.arange(len(species), dtype=np.int32))
    return nodes


def build_positive_edges(
    detections: pd.DataFrame,
    checklists: gpd.GeoDataFrame,
    species: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> pd.DataFrame:
    checklist_index = pd.Series(
        np.arange(len(checklists), dtype=np.int32),
        index=checklists["sampling_event_identifier"],
    )
    species_index = pd.Series(
        np.arange(len(species), dtype=np.int32),
        index=species["species_key"],
    )
    edges = detections[["sampling_event_identifier", "species_key"]].drop_duplicates()
    edges["checklist_index"] = edges["sampling_event_identifier"].map(checklist_index)
    edges["species_index"] = edges["species_key"].map(species_index)
    edges = edges.dropna(subset=["checklist_index", "species_index"]).copy()
    edges["checklist_index"] = edges["checklist_index"].astype(np.int32)
    edges["species_index"] = edges["species_index"].astype(np.int32)
    split = np.full(len(edges), "unused", dtype=object)
    checklist_values = edges["checklist_index"].to_numpy()
    split[train_mask[checklist_values]] = "train"
    split[test_mask[checklist_values]] = "test"
    edges["split"] = split
    edges["label"] = np.int8(1)
    return edges[["checklist_index", "species_index", "split", "label"]]


def sample_negative_edges(
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    negative_ratio: float,
    seed: int,
) -> pd.DataFrame:
    if negative_ratio < 0:
        raise ValueError("--negative-ratio must be non-negative.")

    rng = np.random.default_rng(seed)
    species_count = labels.shape[1]
    rows = []
    for checklist_index, observed in enumerate(labels.astype(bool)):
        positive_count = int(observed.sum())
        if positive_count == 0:
            continue
        available = np.flatnonzero(~observed)
        if len(available) == 0:
            continue
        sample_count = min(len(available), int(np.ceil(positive_count * negative_ratio)))
        if sample_count == 0:
            continue
        sampled = rng.choice(available, size=sample_count, replace=False)
        if train_mask[checklist_index]:
            split = "train"
        elif test_mask[checklist_index]:
            split = "test"
        else:
            split = "unused"
        rows.append(
            pd.DataFrame(
                {
                    "checklist_index": np.full(
                        sample_count, checklist_index, dtype=np.int32
                    ),
                    "species_index": sampled.astype(np.int32),
                    "split": split,
                    "label": np.int8(0),
                }
            )
        )

    if not rows:
        return pd.DataFrame(
            columns=["checklist_index", "species_index", "split", "label"]
        )
    return pd.concat(rows, ignore_index=True)


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite to replace files."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    validate_output_dir(output_dir, args.overwrite)

    checklists, detections, species = load_inputs(
        Path(args.processed_dir),
        args.top_species,
        args.max_checklists,
    )
    labels = build_labels(checklists, detections, species)
    features = build_features(checklists, args.feature_set)
    train_mask, test_mask, split_info = make_split(checklists, labels, species, args)
    feature_matrix, feature_stats = standardize_features(features, train_mask)

    checklist_nodes = build_checklist_nodes(checklists, train_mask, test_mask)
    species_nodes = build_species_nodes(species)
    positive_edges = build_positive_edges(
        detections,
        checklists,
        species,
        train_mask,
        test_mask,
    )
    negative_edges = sample_negative_edges(
        labels,
        train_mask,
        test_mask,
        args.negative_ratio,
        args.negative_seed,
    )

    checklist_nodes.to_parquet(output_dir / "checklist_nodes.parquet", index=False)
    species_nodes.to_csv(output_dir / "species_nodes.csv", index=False)
    positive_edges.to_parquet(output_dir / "positive_edges.parquet", index=False)
    negative_edges.to_parquet(output_dir / "negative_edges.parquet", index=False)
    np.save(output_dir / "checklist_features.npy", feature_matrix)

    metadata = {
        "processed_dir": args.processed_dir,
        "top_species": args.top_species,
        "feature_set": args.feature_set,
        "feature_names": list(features.columns),
        **feature_stats,
        "split": split_info,
        "negative_sampling": {
            "negative_ratio": args.negative_ratio,
            "negative_seed": args.negative_seed,
        },
        "counts": {
            "checklists": int(len(checklists)),
            "species": int(len(species)),
            "train_checklists": int(train_mask.sum()),
            "test_checklists": int(test_mask.sum()),
            "positive_edges": int(len(positive_edges)),
            "negative_edges": int(len(negative_edges)),
            "train_positive_edges": int((positive_edges["split"] == "train").sum()),
            "test_positive_edges": int((positive_edges["split"] == "test").sum()),
            "train_negative_edges": int((negative_edges["split"] == "train").sum()),
            "test_negative_edges": int((negative_edges["split"] == "test").sum()),
        },
        "files": {
            "checklist_nodes": "checklist_nodes.parquet",
            "checklist_features": "checklist_features.npy",
            "species_nodes": "species_nodes.csv",
            "positive_edges": "positive_edges.parquet",
            "negative_edges": "negative_edges.parquet",
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Graph dataset written:")
    for key, value in metadata["counts"].items():
        print(f"  {key}: {value:,}")
    print(f"  output_dir: {output_dir}")


if __name__ == "__main__":
    main()
