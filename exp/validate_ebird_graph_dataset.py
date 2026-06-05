"""
Validate graph-ready eBird dataset exports.

Run from the project root:

    python exp/validate_ebird_graph_dataset.py --graph-dir data/ebird/graph_top100_spatial
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_FILES = [
    "checklist_nodes.parquet",
    "checklist_features.npy",
    "species_nodes.csv",
    "positive_edges.parquet",
    "negative_edges.parquet",
    "metadata.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an eBird graph dataset.")
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Graph dataset directory produced by build_ebird_graph_dataset.py.",
    )
    return parser.parse_args()


def assert_condition(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def edge_key_frame(edges: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(edges[["checklist_index", "species_index"]])


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    missing = [name for name in REQUIRED_FILES if not (graph_dir / name).exists()]
    assert_condition(not missing, f"Missing required files: {', '.join(missing)}")

    metadata = json.loads((graph_dir / "metadata.json").read_text(encoding="utf-8"))
    checklist_nodes = pd.read_parquet(graph_dir / "checklist_nodes.parquet")
    species_nodes = pd.read_csv(graph_dir / "species_nodes.csv")
    positive_edges = pd.read_parquet(graph_dir / "positive_edges.parquet")
    negative_edges = pd.read_parquet(graph_dir / "negative_edges.parquet")
    features = np.load(graph_dir / "checklist_features.npy")

    counts = metadata["counts"]
    assert_condition(
        len(checklist_nodes) == counts["checklists"],
        "Checklist count does not match metadata.",
    )
    assert_condition(
        len(species_nodes) == counts["species"],
        "Species count does not match metadata.",
    )
    assert_condition(
        len(positive_edges) == counts["positive_edges"],
        "Positive edge count does not match metadata.",
    )
    assert_condition(
        len(negative_edges) == counts["negative_edges"],
        "Negative edge count does not match metadata.",
    )
    assert_condition(
        features.shape[0] == len(checklist_nodes),
        "Feature matrix row count does not match checklist nodes.",
    )
    assert_condition(
        features.shape[1] == len(metadata["feature_names"]),
        "Feature matrix column count does not match feature names.",
    )
    assert_condition(np.isfinite(features).all(), "Feature matrix contains NaN or inf.")

    train_mask = checklist_nodes["train_mask"].astype(bool).to_numpy()
    test_mask = checklist_nodes["test_mask"].astype(bool).to_numpy()
    assert_condition(
        not np.any(train_mask & test_mask),
        "At least one checklist is marked both train and test.",
    )
    assert_condition(train_mask.any(), "No train checklists found.")
    assert_condition(test_mask.any(), "No test checklists found.")
    assert_condition(
        int(train_mask.sum()) == counts["train_checklists"],
        "Train checklist count does not match metadata.",
    )
    assert_condition(
        int(test_mask.sum()) == counts["test_checklists"],
        "Test checklist count does not match metadata.",
    )

    species_count = len(species_nodes)
    checklist_count = len(checklist_nodes)
    for name, edges, label in [
        ("positive", positive_edges, 1),
        ("negative", negative_edges, 0),
    ]:
        required_columns = {"checklist_index", "species_index", "split", "label"}
        assert_condition(
            required_columns.issubset(edges.columns),
            f"{name} edges are missing required columns.",
        )
        assert_condition(
            (edges["label"] == label).all(),
            f"{name} edges have unexpected labels.",
        )
        assert_condition(
            edges["checklist_index"].between(0, checklist_count - 1).all(),
            f"{name} edges contain invalid checklist_index values.",
        )
        assert_condition(
            edges["species_index"].between(0, species_count - 1).all(),
            f"{name} edges contain invalid species_index values.",
        )
        valid_split = edges["split"].isin(["train", "test", "unused"]).all()
        assert_condition(valid_split, f"{name} edges contain invalid split labels.")

        checklist_index = edges["checklist_index"].to_numpy()
        split = edges["split"].to_numpy()
        assert_condition(
            np.all(split[train_mask[checklist_index]] == "train"),
            f"{name} train checklist edges have wrong split labels.",
        )
        assert_condition(
            np.all(split[test_mask[checklist_index]] == "test"),
            f"{name} test checklist edges have wrong split labels.",
        )

    duplicate_positive = positive_edges.duplicated(
        ["checklist_index", "species_index"]
    ).sum()
    duplicate_negative = negative_edges.duplicated(
        ["checklist_index", "species_index"]
    ).sum()
    assert_condition(duplicate_positive == 0, "Duplicate positive edges found.")
    assert_condition(duplicate_negative == 0, "Duplicate negative edges found.")

    overlap = edge_key_frame(positive_edges).intersection(edge_key_frame(negative_edges))
    assert_condition(len(overlap) == 0, "Positive and negative edges overlap.")

    split_summary = pd.concat(
        [
            positive_edges.assign(edge_type="positive"),
            negative_edges.assign(edge_type="negative"),
        ],
        ignore_index=True,
    ).groupby(["split", "edge_type"]).size()

    print("Graph dataset validation passed.")
    print(f"Graph directory: {graph_dir}")
    print(f"Feature matrix: {features.shape[0]:,} x {features.shape[1]:,}")
    print("Counts:")
    for key, value in counts.items():
        print(f"  {key}: {value:,}")
    print("Split edge counts:")
    print(split_summary.to_string())


if __name__ == "__main__":
    main()
