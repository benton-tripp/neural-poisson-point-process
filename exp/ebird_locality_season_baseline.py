"""
Fit simple aggregated locality-season baselines.

This is the first modeling step after building locality-season replication
tables. It treats each locality-season/species row as a binomial observation:

    n_detections ~ Binomial(n_checklists, p_species_locality_season)

The goal is not a full latent occupancy model yet. Instead, it compares simple
availability-style features (environment + season) with effort-summary features
and their additive combination.

Run from the project root:

    python exp/ebird_locality_season_baseline.py --dataset-dir data/ebird/locality_season_top100 --epochs 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ebird_joint_tabular_baseline import auc_roc, average_precision


DEFAULT_DATASET_DIR = "data/ebird/locality_season_top100"
DEFAULT_OUTPUT_DIR_NAME = "baselines"
SEED = 19

AVAILABILITY_NUMERIC_COLUMNS = [
    "canopy_median",
    "elevation_median",
    "distance_to_waterbody_m_median",
    "distance_to_coastline_m_median",
]
EFFORT_NUMERIC_COLUMNS = [
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
]
LOG1P_COLUMNS = {
    "distance_to_waterbody_m_median",
    "distance_to_coastline_m_median",
    "n_dates",
    "unique_observers",
    "duration_mean",
    "duration_median",
    "duration_p90",
    "effort_distance_mean",
    "effort_distance_median",
    "effort_distance_p90",
    "number_observers_mean",
}
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
        description="Fit simple binomial locality-season detection baselines."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Locality-season dataset directory. Defaults to {DEFAULT_DATASET_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to dataset-dir/baselines.",
    )
    parser.add_argument(
        "--test-season-year",
        type=int,
        default=2023,
        help="Season year held out for testing. Defaults to 2023.",
    )
    parser.add_argument(
        "--include-inadequate",
        action="store_true",
        help="Include locality-seasons that failed the adequate_sampling flag.",
    )
    parser.add_argument(
        "--include-coordinates",
        action="store_true",
        help="Add x/y coordinates to the availability feature set.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Training epochs for each model. Defaults to 30.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Training batch size. Defaults to 65,536.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
        help="AdamW learning rate. Defaults to 1e-2.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay. Defaults to 1e-4.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for smoke tests. Defaults to all rows.",
    )
    parser.add_argument(
        "--focus-species",
        nargs="*",
        default=DEFAULT_FOCUS_SPECIES,
        help="Common names included in the focus species season output.",
    )
    return parser.parse_args()


class SpeciesLinearBinomialModel(nn.Module):
    def __init__(
        self,
        feature_count: int,
        species_count: int,
        initial_species_logits: np.ndarray,
    ) -> None:
        super().__init__()
        self.weights = nn.Embedding(species_count, feature_count)
        self.bias = nn.Embedding(species_count, 1)
        nn.init.zeros_(self.weights.weight)
        with torch.no_grad():
            self.bias.weight[:, 0] = torch.from_numpy(
                initial_species_logits.astype(np.float32)
            )

    def forward(self, features: torch.Tensor, species_index: torch.Tensor) -> torch.Tensor:
        weights = self.weights(species_index)
        bias = self.bias(species_index).squeeze(-1)
        return (features * weights).sum(dim=1) + bias


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def load_triplets(dataset_dir: Path, include_inadequate: bool, max_rows: int | None) -> pd.DataFrame:
    columns = sorted(
        set(
            [
                "locality_season_id",
                "species_index",
                "species_key",
                "common_name",
                "scientific_name",
                "season_year",
                "season_name",
                "n_checklists",
                "n_detections",
                "adequate_sampling",
                "naive_detection_rate",
                "x",
                "y",
            ]
            + AVAILABILITY_NUMERIC_COLUMNS
            + EFFORT_NUMERIC_COLUMNS
        )
    )
    frame = pd.read_parquet(dataset_dir / "locality_season_species.parquet", columns=columns)
    if not include_inadequate:
        frame = frame.loc[frame["adequate_sampling"]].copy()
    if max_rows is not None:
        frame = frame.head(max_rows).copy()
    frame["season_name"] = frame["season_name"].astype(str)
    return frame.reset_index(drop=True)


def add_one_hot(frame: pd.DataFrame, column: str, prefix: str) -> pd.DataFrame:
    values = pd.get_dummies(frame[column].astype(str), prefix=prefix, dtype=float)
    return values.sort_index(axis=1)


def numeric_features(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    features = pd.DataFrame(index=frame.index)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").astype(float)
        values = values.fillna(values.median())
        if column in LOG1P_COLUMNS:
            values = np.log1p(values.clip(lower=0.0))
        features[column] = values
    return features


def build_feature_frame(frame: pd.DataFrame, feature_set: str, include_coordinates: bool) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    if feature_set in {"availability", "both"}:
        parts.append(add_one_hot(frame, "season_name", "season"))
        year = pd.to_numeric(frame["season_year"], errors="coerce").astype(float)
        parts.append(pd.DataFrame({"season_year": year}, index=frame.index))
        columns = AVAILABILITY_NUMERIC_COLUMNS.copy()
        if include_coordinates:
            columns.extend(["x", "y"])
        parts.append(numeric_features(frame, columns))
    if feature_set in {"effort", "both"}:
        parts.append(numeric_features(frame, EFFORT_NUMERIC_COLUMNS))
    if not parts:
        raise ValueError(f"Unknown feature set: {feature_set}")
    return pd.concat(parts, axis=1)


def standardize(
    features: pd.DataFrame,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    values = features.to_numpy(dtype=np.float32)
    train = values[train_mask]
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    scaled = (values - mean) / std
    return scaled.astype(np.float32), {
        "feature_names": list(features.columns),
        "feature_mean": mean.ravel().astype(float).tolist(),
        "feature_std": std.ravel().astype(float).tolist(),
    }


def make_split(frame: pd.DataFrame, test_season_year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    years = frame["season_year"].astype(int).to_numpy()
    train_mask = years < test_season_year
    test_mask = years == test_season_year
    unused_mask = years > test_season_year
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            f"Temporal split failed for test season year {test_season_year}: "
            f"train={train_mask.sum()}, test={test_mask.sum()}."
        )
    return train_mask, test_mask, unused_mask


def prevalence_scores(
    species_index: np.ndarray,
    n_detections: np.ndarray,
    n_checklists: np.ndarray,
    train_mask: np.ndarray,
    smoothing: float = 1.0,
) -> np.ndarray:
    species_count = int(species_index.max()) + 1
    detections = np.bincount(
        species_index[train_mask],
        weights=n_detections[train_mask],
        minlength=species_count,
    )
    trials = np.bincount(
        species_index[train_mask],
        weights=n_checklists[train_mask],
        minlength=species_count,
    )
    rates = (detections + smoothing) / (trials + 2.0 * smoothing)
    return rates[species_index]


def train_species_logits(
    species_index: np.ndarray,
    n_detections: np.ndarray,
    n_checklists: np.ndarray,
    train_mask: np.ndarray,
    smoothing: float = 1.0,
) -> np.ndarray:
    species_count = int(species_index.max()) + 1
    detections = np.bincount(
        species_index[train_mask],
        weights=n_detections[train_mask],
        minlength=species_count,
    )
    trials = np.bincount(
        species_index[train_mask],
        weights=n_checklists[train_mask],
        minlength=species_count,
    )
    rates = (detections + smoothing) / (trials + 2.0 * smoothing)
    rates = np.clip(rates, 1e-5, 1.0 - 1e-5)
    return np.log(rates / (1.0 - rates)).astype(np.float32)


def fit_model(
    features: np.ndarray,
    species_index: np.ndarray,
    target_rate: np.ndarray,
    n_checklists: np.ndarray,
    train_mask: np.ndarray,
    args: argparse.Namespace,
    model_name: str,
    initial_species_logits: np.ndarray,
) -> np.ndarray:
    torch.manual_seed(SEED)
    species_count = int(species_index.max()) + 1
    model = SpeciesLinearBinomialModel(
        features.shape[1], species_count, initial_species_logits
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    dataset = TensorDataset(
        torch.from_numpy(features[train_mask].astype(np.float32)),
        torch.from_numpy(species_index[train_mask].astype(np.int64)),
        torch.from_numpy(target_rate[train_mask].astype(np.float32)),
        torch.from_numpy(n_checklists[train_mask].astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, sb, yb, nb in loader:
            optimizer.zero_grad()
            logits = model(xb, sb)
            loss_raw = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, reduction="none"
            )
            loss = (loss_raw * nb).sum() / nb.sum().clamp_min(1.0)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            print(f"{model_name} epoch {epoch:>4}: train weighted BCE={np.mean(losses):.5f}")

    model.eval()
    scores = np.empty(len(features), dtype=np.float32)
    eval_dataset = TensorDataset(
        torch.from_numpy(features.astype(np.float32)),
        torch.from_numpy(species_index.astype(np.int64)),
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    offset = 0
    with torch.no_grad():
        for xb, sb in eval_loader:
            logits = model(xb, sb).numpy()
            batch_size = len(logits)
            scores[offset : offset + batch_size] = sigmoid(logits)
            offset += batch_size
    return scores


def weighted_bce(k: np.ndarray, n: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-7
    p = np.clip(p, eps, 1.0 - eps)
    losses = -(k * np.log(p) + (n - k) * np.log1p(-p))
    return float(losses.sum() / np.maximum(n.sum(), 1.0))


def summarize_overall(
    frame: pd.DataFrame,
    mask: np.ndarray,
    scores: np.ndarray,
    model_name: str,
) -> dict:
    k = frame.loc[mask, "n_detections"].to_numpy(dtype=float)
    n = frame.loc[mask, "n_checklists"].to_numpy(dtype=float)
    y_binary = k > 0
    p = scores[mask]
    observed_rate = float(k.sum() / n.sum())
    predicted_rate = float((p * n).sum() / n.sum())
    rate = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    return {
        "model": model_name,
        "rows": int(mask.sum()),
        "trials": int(n.sum()),
        "detections": int(k.sum()),
        "observed_detection_rate": observed_rate,
        "mean_predicted_detection_rate": predicted_rate,
        "calibration_error": abs(predicted_rate - observed_rate),
        "weighted_bce": weighted_bce(k, n, p),
        "weighted_mae_rate": float((np.abs(p - rate) * n).sum() / n.sum()),
        "positive_triplet_auroc": auc_roc(y_binary.astype(float), p),
        "positive_triplet_auprc": average_precision(y_binary.astype(float), p),
    }


def summarize_by_species(
    frame: pd.DataFrame,
    mask: np.ndarray,
    scores: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    work = frame.loc[mask, ["species_key", "common_name", "scientific_name", "n_detections", "n_checklists"]].copy()
    work["score"] = scores[mask]
    rows = []
    for (species_key, common_name, scientific_name), group in work.groupby(
        ["species_key", "common_name", "scientific_name"], observed=True
    ):
        k = group["n_detections"].to_numpy(dtype=float)
        n = group["n_checklists"].to_numpy(dtype=float)
        p = group["score"].to_numpy(dtype=float)
        y_binary = k > 0
        rows.append(
            {
                "model": model_name,
                "species_key": species_key,
                "common_name": common_name,
                "scientific_name": scientific_name,
                "rows": int(len(group)),
                "trials": int(n.sum()),
                "detections": int(k.sum()),
                "positive_locality_seasons": int(y_binary.sum()),
                "observed_detection_rate": float(k.sum() / n.sum()),
                "mean_predicted_detection_rate": float((p * n).sum() / n.sum()),
                "calibration_error": abs(float((p * n).sum() / n.sum()) - float(k.sum() / n.sum())),
                "weighted_bce": weighted_bce(k, n, p),
                "positive_triplet_auroc": auc_roc(y_binary.astype(float), p),
                "positive_triplet_auprc": average_precision(y_binary.astype(float), p),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_species_season(
    frame: pd.DataFrame,
    mask: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    focus_species: list[str],
) -> pd.DataFrame:
    base_columns = [
        "common_name",
        "season_name",
        "n_detections",
        "n_checklists",
    ]
    work = frame.loc[mask & frame["common_name"].isin(focus_species), base_columns].copy()
    if work.empty:
        return pd.DataFrame()
    for model_name, scores in scores_by_model.items():
        work[f"{model_name}_score"] = scores[mask & frame["common_name"].isin(focus_species)]

    rows = []
    for (common_name, season_name), group in work.groupby(["common_name", "season_name"], observed=True):
        k = group["n_detections"].to_numpy(dtype=float)
        n = group["n_checklists"].to_numpy(dtype=float)
        row = {
            "common_name": common_name,
            "season_name": season_name,
            "rows": int(len(group)),
            "positive_locality_seasons": int((k > 0).sum()),
            "trials": int(n.sum()),
            "detections": int(k.sum()),
            "observed_detection_rate": float(k.sum() / n.sum()),
        }
        for model_name in scores_by_model:
            p = group[f"{model_name}_score"].to_numpy(dtype=float)
            predicted = float((p * n).sum() / n.sum())
            row[f"{model_name}_predicted_detection_rate"] = predicted
            row[f"{model_name}_calibration_error"] = abs(
                predicted - row["observed_detection_rate"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["common_name", "season_name"])


def build_and_fit_models(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    species_index = frame["species_index"].to_numpy(dtype=np.int64)
    n_checklists = frame["n_checklists"].to_numpy(dtype=np.float32)
    n_detections = frame["n_detections"].to_numpy(dtype=np.float32)
    target_rate = np.divide(
        n_detections,
        n_checklists,
        out=np.zeros_like(n_detections),
        where=n_checklists > 0,
    ).astype(np.float32)

    scores = {
        "train_prevalence": prevalence_scores(
            species_index, n_detections, n_checklists, train_mask
        ).astype(np.float32)
    }
    initial_species_logits = train_species_logits(
        species_index, n_detections, n_checklists, train_mask
    )
    feature_metadata: dict[str, dict] = {}
    for model_name, feature_set in [
        ("availability", "availability"),
        ("effort", "effort"),
        ("combined", "both"),
    ]:
        features = build_feature_frame(frame, feature_set, args.include_coordinates)
        x, metadata = standardize(features, train_mask)
        scores[model_name] = fit_model(
            x,
            species_index,
            target_rate,
            n_checklists,
            train_mask,
            args,
            model_name,
            initial_species_logits,
        )
        feature_metadata[model_name] = metadata
    return scores, feature_metadata


def write_outputs(
    output_dir: Path,
    overall: pd.DataFrame,
    species_metrics: pd.DataFrame,
    focus: pd.DataFrame,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(output_dir / "locality_season_baseline_metrics.csv", index=False)
    species_metrics.to_csv(
        output_dir / "locality_season_baseline_species_metrics.csv", index=False
    )
    focus.to_csv(
        output_dir / "locality_season_baseline_focus_species_season.csv", index=False
    )
    (output_dir / "locality_season_baseline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else dataset_dir / DEFAULT_OUTPUT_DIR_NAME
    )
    frame = load_triplets(dataset_dir, args.include_inadequate, args.max_rows)
    train_mask, test_mask, unused_mask = make_split(frame, args.test_season_year)
    print(
        "Rows: "
        f"train={int(train_mask.sum()):,}, test={int(test_mask.sum()):,}, "
        f"unused={int(unused_mask.sum()):,}"
    )
    print(
        "Trials: "
        f"train={int(frame.loc[train_mask, 'n_checklists'].sum()):,}, "
        f"test={int(frame.loc[test_mask, 'n_checklists'].sum()):,}"
    )

    scores, feature_metadata = build_and_fit_models(frame, train_mask, args)
    overall_rows = []
    species_frames = []
    for model_name, model_scores in scores.items():
        overall_rows.append(summarize_overall(frame, test_mask, model_scores, model_name))
        species_frames.append(summarize_by_species(frame, test_mask, model_scores, model_name))
    overall = pd.DataFrame(overall_rows)
    species_metrics = pd.concat(species_frames, ignore_index=True)
    focus = summarize_by_species_season(frame, test_mask, scores, args.focus_species)

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "test_season_year": int(args.test_season_year),
        "include_inadequate": bool(args.include_inadequate),
        "include_coordinates": bool(args.include_coordinates),
        "rows": {
            "train": int(train_mask.sum()),
            "test": int(test_mask.sum()),
            "unused": int(unused_mask.sum()),
        },
        "trials": {
            "train": int(frame.loc[train_mask, "n_checklists"].sum()),
            "test": int(frame.loc[test_mask, "n_checklists"].sum()),
        },
        "models": overall.set_index("model").to_dict(orient="index"),
        "features": feature_metadata,
    }
    write_outputs(output_dir, overall, species_metrics, focus, summary)

    print()
    print("Locality-season baseline metrics:")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print()
    print("Focus species/season calibration:")
    display_cols = [
        "common_name",
        "season_name",
        "positive_locality_seasons",
        "observed_detection_rate",
        "availability_predicted_detection_rate",
        "effort_predicted_detection_rate",
        "combined_predicted_detection_rate",
    ]
    if not focus.empty:
        print(focus[display_cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print()
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
