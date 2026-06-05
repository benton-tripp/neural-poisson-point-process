"""
Top-species checklist prediction baselines for processed eBird bulk data.

Run from the project root:

    python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 20 --feature-set both --epochs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 19
DEFAULT_PROCESSED_DIR = "data/ebird/processed_nc_2020_2023"
DEFAULT_OUTPUT_DIR = "data/ebird/baselines"
ECOLOGY_COLUMNS = [
    "canopy_median",
    "nc_usgs30m_match_tcc",
    "distance_to_waterbody_m",
    "distance_to_coastline_m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a small tabular multi-species eBird checklist baseline."
    )
    parser.add_argument(
        "--processed-dir",
        default=DEFAULT_PROCESSED_DIR,
        help=f"Directory with checklists.geoparquet, detections.parquet, and species.csv. Defaults to {DEFAULT_PROCESSED_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for baseline CSV/JSON outputs. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--top-species",
        type=int,
        default=20,
        help="Number of most frequently detected species to model. Defaults to 20.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["effort", "ecology", "both"],
        default="both",
        help="Checklist features used by the model. Defaults to both.",
    )
    parser.add_argument(
        "--model",
        choices=["linear", "mlp"],
        default="linear",
        help="Model family to fit. Defaults to linear.",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        default=2023,
        help="Year held out for testing. Defaults to 2023.",
    )
    parser.add_argument(
        "--split",
        choices=["temporal", "spatial-stratified"],
        default="temporal",
        help="Train/test split strategy. Defaults to temporal.",
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
        help="Number of common species included in spatial split balancing. Defaults to 20.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=SEED,
        help="Random seed used to break ties in spatial split selection. Defaults to 19.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Training epochs for the linear multi-label model. Defaults to 50.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8192,
        help="Mini-batch size. Defaults to 8192.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
        help="Adam learning rate. Defaults to 1e-2.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW-style L2 penalty through optimizer weight decay. Defaults to 1e-4.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden layer width for --model mlp. Defaults to 64.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=1,
        help="Number of hidden layers for --model mlp. Defaults to 1.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help="Dropout for --model mlp. Defaults to 0.10.",
    )
    parser.add_argument(
        "--max-checklists",
        type=int,
        default=None,
        help="Optional row limit for smoke tests. Defaults to all checklists.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of predicted-probability bins for calibration output. Defaults to 10.",
    )
    return parser.parse_args()


def auc_roc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    positives = int(y_true.sum())
    negatives = int((~y_true).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores, method="average")
    positive_rank_sum = ranks[y_true].sum()
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )
    return float(auc)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(bool)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")[::-1]
    y_sorted = y_true[order]
    precision = np.cumsum(y_sorted) / (np.arange(len(y_sorted)) + 1)
    return float(precision[y_sorted].sum() / positives)


def load_inputs(processed_dir: Path, top_species: int, max_checklists: int | None):
    checklists = gpd.read_parquet(processed_dir / "checklists.geoparquet")
    if max_checklists is not None:
        checklists = checklists.head(max_checklists).copy()

    species = pd.read_csv(processed_dir / "species.csv").head(top_species).copy()
    detections = pd.read_parquet(
        processed_dir / "detections.parquet",
        columns=["sampling_event_identifier", "species_key"],
    )
    detections = detections[detections["species_key"].isin(species["species_key"])]
    detections = detections[
        detections["sampling_event_identifier"].isin(
            checklists["sampling_event_identifier"]
        )
    ]
    return checklists, detections, species


def build_labels(
    checklists: gpd.GeoDataFrame,
    detections: pd.DataFrame,
    species: pd.DataFrame,
) -> np.ndarray:
    checklist_index = pd.Series(
        np.arange(len(checklists), dtype=np.int64),
        index=checklists["sampling_event_identifier"],
    )
    species_index = pd.Series(
        np.arange(len(species), dtype=np.int64),
        index=species["species_key"],
    )
    rows = detections["sampling_event_identifier"].map(checklist_index).to_numpy()
    cols = detections["species_key"].map(species_index).to_numpy()
    valid = (~pd.isna(rows)) & (~pd.isna(cols))

    labels = np.zeros((len(checklists), len(species)), dtype=np.float32)
    labels[rows[valid].astype(np.int64), cols[valid].astype(np.int64)] = 1.0
    return labels


def add_cyclic_feature(frame: pd.DataFrame, column: str, period: float) -> pd.DataFrame:
    values = frame[column].astype(float).to_numpy()
    radians = 2.0 * np.pi * values / period
    return pd.DataFrame(
        {
            f"{column}_sin": np.sin(radians),
            f"{column}_cos": np.cos(radians),
        },
        index=frame.index,
    )


def build_features(checklists: gpd.GeoDataFrame, feature_set: str) -> pd.DataFrame:
    features = pd.DataFrame(index=checklists.index)
    features["x"] = checklists.geometry.x
    features["y"] = checklists.geometry.y
    features = pd.concat(
        [
            features,
            add_cyclic_feature(checklists, "day_of_year", 366.0),
            add_cyclic_feature(checklists, "day_of_week", 7.0),
        ],
        axis=1,
    )

    if feature_set in {"effort", "both"}:
        effort_distance = checklists["effort_distance_km"].fillna(0.0)
        features["duration_log1p"] = np.log1p(checklists["duration_minutes"])
        features["effort_distance_log1p"] = np.log1p(effort_distance)
        features["number_observers_log1p"] = np.log1p(checklists["number_observers"])
        features["is_traveling"] = (checklists["protocol_code"] == "P22").astype(float)

    if feature_set in {"ecology", "both"}:
        missing = [col for col in ECOLOGY_COLUMNS if col not in checklists.columns]
        if missing:
            raise ValueError(f"Missing ecology columns: {', '.join(missing)}")
        for col in ECOLOGY_COLUMNS:
            values = checklists[col].astype(float)
            if col.startswith("distance_to_"):
                values = np.log1p(values)
            features[col] = values

    return features.astype(np.float32)


def standardize_train_test(
    features: pd.DataFrame,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = features.loc[train_mask].to_numpy(dtype=np.float32)
    test = features.loc[test_mask].to_numpy(dtype=np.float32)
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (train - mean) / std, (test - mean) / std


def assign_spatial_blocks(
    checklists: gpd.GeoDataFrame,
    blocks_per_dim: int,
) -> np.ndarray:
    if blocks_per_dim <= 1:
        raise ValueError("--spatial-blocks-per-dim must be greater than 1.")
    x = checklists.geometry.x.to_numpy(dtype=np.float64)
    y = checklists.geometry.y.to_numpy(dtype=np.float64)
    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    if x_span <= 0 or y_span <= 0:
        raise ValueError("Spatial split requires non-degenerate point coordinates.")
    x_bin = np.floor((x - x.min()) / x_span * blocks_per_dim).astype(np.int64)
    y_bin = np.floor((y - y.min()) / y_span * blocks_per_dim).astype(np.int64)
    x_bin = np.clip(x_bin, 0, blocks_per_dim - 1)
    y_bin = np.clip(y_bin, 0, blocks_per_dim - 1)
    return y_bin * blocks_per_dim + x_bin


def spatial_stratification_frame(
    checklists: gpd.GeoDataFrame,
    labels: np.ndarray,
    species: pd.DataFrame,
    stratify_species_count: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=checklists.index)
    frame["duration_log1p"] = np.log1p(checklists["duration_minutes"].astype(float))
    frame["effort_distance_log1p"] = np.log1p(
        checklists["effort_distance_km"].fillna(0.0).astype(float)
    )
    frame["number_observers_log1p"] = np.log1p(
        checklists["number_observers"].astype(float)
    )
    frame["is_traveling"] = (checklists["protocol_code"] == "P22").astype(float)
    for col in ECOLOGY_COLUMNS:
        if col not in checklists.columns:
            raise ValueError(f"Missing ecology column for spatial stratification: {col}")
        values = checklists[col].astype(float)
        if col.startswith("distance_to_"):
            values = np.log1p(values)
        frame[col] = values
    species_count = min(stratify_species_count, labels.shape[1], len(species))
    for idx in range(species_count):
        key = species.iloc[idx]["species_key"]
        frame[f"species_{key}"] = labels[:, idx]
    return frame.astype(np.float64)


def select_spatial_test_blocks(
    block_ids: np.ndarray,
    stratify_values: pd.DataFrame,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    unique_blocks = np.array(sorted(np.unique(block_ids)))
    if len(unique_blocks) < 2:
        raise ValueError("Spatial split requires at least two populated blocks.")

    values = stratify_values.to_numpy(dtype=np.float64)
    counts = np.array([(block_ids == block).sum() for block in unique_blocks])
    sums = np.vstack([values[block_ids == block].sum(axis=0) for block in unique_blocks])
    total_count = counts.sum()
    target_count = total_count * test_fraction
    target_mean = sums.sum(axis=0) / total_count
    row_std = values.std(axis=0)
    row_std[row_std == 0.0] = 1.0

    selected: list[int] = []
    selected_count = 0
    selected_sum = np.zeros(sums.shape[1], dtype=np.float64)
    remaining = list(range(len(unique_blocks)))

    while selected_count < target_count and remaining:
        best_pos = None
        best_score = np.inf
        rng.shuffle(remaining)
        for pos in remaining:
            candidate_count = selected_count + counts[pos]
            candidate_mean = (selected_sum + sums[pos]) / candidate_count
            mean_score = np.mean(np.abs((candidate_mean - target_mean) / row_std))
            size_score = abs(candidate_count - target_count) / total_count
            score = mean_score + 2.0 * size_score
            if score < best_score:
                best_score = score
                best_pos = pos
        selected.append(best_pos)
        selected_count += counts[best_pos]
        selected_sum += sums[best_pos]
        remaining.remove(best_pos)

    selected_blocks = unique_blocks[np.array(selected, dtype=np.int64)]
    test_mask = np.isin(block_ids, selected_blocks)
    if test_mask.all() or not test_mask.any():
        raise ValueError("Spatial split failed to create both train and test rows.")

    selected_mean = selected_sum / selected_count
    balance = {
        "blocks_total": int(len(unique_blocks)),
        "test_blocks": int(len(selected_blocks)),
        "test_fraction_actual": float(test_mask.mean()),
        "test_blocks_ids": [int(block) for block in selected_blocks],
        "mean_absolute_standardized_balance_error": float(
            np.mean(np.abs((selected_mean - target_mean) / row_std))
        ),
    }
    return test_mask, balance


def make_split(
    checklists: gpd.GeoDataFrame,
    labels: np.ndarray,
    species: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if args.split == "temporal":
        train_mask = checklists["year"].to_numpy() != args.test_year
        test_mask = checklists["year"].to_numpy() == args.test_year
        split_info = {"split": "temporal", "test_year": args.test_year}
    else:
        block_ids = assign_spatial_blocks(checklists, args.spatial_blocks_per_dim)
        stratify_values = spatial_stratification_frame(
            checklists,
            labels,
            species,
            args.stratify_species_count,
        )
        test_mask, balance = select_spatial_test_blocks(
            block_ids,
            stratify_values,
            args.test_fraction,
            args.split_seed,
        )
        train_mask = ~test_mask
        split_info = {
            "split": "spatial-stratified",
            "spatial_blocks_per_dim": args.spatial_blocks_per_dim,
            "requested_test_fraction": args.test_fraction,
            "stratify_species_count": args.stratify_species_count,
            **balance,
        }

    if not train_mask.any() or not test_mask.any():
        raise ValueError(f"Split {args.split} did not create train/test rows.")
    return train_mask, test_mask, split_info


class LinearChecklistModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPChecklistModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        dropout: float,
    ):
        super().__init__()
        if hidden_layers <= 0:
            raise ValueError("--hidden-layers must be greater than zero for MLP.")
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(input_dim: int, output_dim: int, args: argparse.Namespace) -> nn.Module:
    if args.model == "linear":
        return LinearChecklistModel(input_dim, output_dim)
    return MLPChecklistModel(
        input_dim,
        output_dim,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
    )


def model_label(args: argparse.Namespace) -> str:
    return f"{args.model}_{args.feature_set}"


def model_file_suffix(args: argparse.Namespace) -> str:
    return "" if args.model == "linear" else f"_{args.model}"


def fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    torch.manual_seed(SEED)
    model = build_model(x_train.shape[1], y_train.shape[1], args)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(
        torch.from_numpy(x_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            print(f"epoch {epoch:>4}: train BCE={np.mean(losses):.5f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x_test.astype(np.float32))).numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def evaluate(
    y_test: np.ndarray,
    scores: np.ndarray,
    species: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    for idx, row in species.reset_index(drop=True).iterrows():
        y = y_test[:, idx]
        p = scores[:, idx]
        rows.append(
            {
                "model": model_name,
                "species_key": row["species_key"],
                "common_name": row["common_name"],
                "test_prevalence": float(y.mean()),
                "test_detections": int(y.sum()),
                "auroc": auc_roc(y, p),
                "auprc": average_precision(y, p),
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame, y_test: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "macro_auroc": float(metrics["auroc"].mean()),
        "macro_auprc": float(metrics["auprc"].mean()),
        "micro_auroc": auc_roc(y_test.ravel(), scores.ravel()),
        "micro_auprc": average_precision(y_test.ravel(), scores.ravel()),
    }


def calibration_row(
    model_name: str,
    calibration_type: str,
    stratum: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    checklist_count: int | None = None,
) -> dict:
    y_flat = y_true.ravel()
    score_flat = scores.ravel()
    if len(y_flat) == 0:
        mean_predicted = float("nan")
        observed_rate = float("nan")
        error = float("nan")
    else:
        mean_predicted = float(score_flat.mean())
        observed_rate = float(y_flat.mean())
        error = abs(mean_predicted - observed_rate)
    return {
        "model": model_name,
        "calibration_type": calibration_type,
        "stratum": stratum,
        "pairs": int(len(y_flat)),
        "checklists": checklist_count,
        "mean_predicted": mean_predicted,
        "observed_rate": observed_rate,
        "calibration_error": error,
    }


def probability_bin_calibration(
    model_name: str,
    y_test: np.ndarray,
    scores: np.ndarray,
    bins: int,
) -> list[dict]:
    if bins <= 0:
        raise ValueError("--calibration-bins must be greater than zero.")
    y_flat = y_test.ravel()
    score_flat = scores.ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.digitize(score_flat, edges[1:-1], right=False)

    rows = []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        lower = edges[bin_id]
        upper = edges[bin_id + 1]
        bracket = "[" if bin_id == 0 else "("
        rows.append(
            calibration_row(
                model_name,
                "predicted_probability_bin",
                f"{bracket}{lower:.1f}, {upper:.1f}]",
                y_flat[mask].reshape(-1, 1),
                score_flat[mask].reshape(-1, 1),
                None,
            )
        )
    return rows


def effort_strata(checklists_test: gpd.GeoDataFrame) -> dict[str, pd.Series]:
    duration = checklists_test["duration_minutes"].astype(float)
    distance = checklists_test["effort_distance_km"].fillna(0.0).astype(float)
    observers = checklists_test["number_observers"].astype(float)
    return {
        "protocol": checklists_test["protocol_code"].astype(str),
        "duration_minutes": pd.cut(
            duration,
            bins=[0, 10, 30, 60, 120, np.inf],
            labels=["1-10", "11-30", "31-60", "61-120", "121+"],
            include_lowest=True,
        ),
        "effort_distance_km": pd.cut(
            distance,
            bins=[-0.001, 0, 0.5, 2, 5, np.inf],
            labels=["0", "(0,0.5]", "(0.5,2]", "(2,5]", "5+"],
            include_lowest=True,
        ),
        "number_observers": pd.cut(
            observers,
            bins=[0, 1, 2, np.inf],
            labels=["1", "2", "3+"],
            include_lowest=True,
        ),
    }


def effort_stratum_calibration(
    model_name: str,
    checklists_test: gpd.GeoDataFrame,
    y_test: np.ndarray,
    scores: np.ndarray,
) -> list[dict]:
    rows = []
    for variable, strata in effort_strata(checklists_test).items():
        for value in sorted(strata.dropna().unique(), key=str):
            mask = (strata == value).to_numpy()
            if not mask.any():
                continue
            rows.append(
                calibration_row(
                    model_name,
                    f"effort_{variable}",
                    str(value),
                    y_test[mask],
                    scores[mask],
                    int(mask.sum()),
                )
            )
    return rows


def build_calibration_table(
    checklists_test: gpd.GeoDataFrame,
    y_test: np.ndarray,
    model_scores: dict[str, np.ndarray],
    bins: int,
) -> pd.DataFrame:
    rows = []
    for model_name, scores in model_scores.items():
        rows.extend(probability_bin_calibration(model_name, y_test, scores, bins))
        rows.extend(effort_stratum_calibration(model_name, checklists_test, y_test, scores))
    return pd.DataFrame(rows)


def summarize_calibration(calibration: pd.DataFrame) -> dict[str, dict]:
    summaries = {}
    probability = calibration[
        calibration["calibration_type"] == "predicted_probability_bin"
    ].copy()
    for model_name, group in probability.groupby("model"):
        weights = group["pairs"].to_numpy(dtype=np.float64)
        errors = group["calibration_error"].to_numpy(dtype=np.float64)
        summaries[model_name] = {
            "expected_calibration_error": float(
                np.average(errors, weights=weights) if weights.sum() > 0 else np.nan
            ),
            "max_probability_bin_error": float(np.nanmax(errors)),
        }
    return summaries


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_dir = Path(args.processed_dir)
    checklists, detections, species = load_inputs(
        processed_dir, args.top_species, args.max_checklists
    )
    labels = build_labels(checklists, detections, species)
    features = build_features(checklists, args.feature_set)

    train_mask, test_mask, split_info = make_split(
        checklists,
        labels,
        species,
        args,
    )

    x_train, x_test = standardize_train_test(features, train_mask, test_mask)
    y_train = labels[train_mask]
    y_test = labels[test_mask]

    prevalence = y_train.mean(axis=0)
    prevalence_scores = np.broadcast_to(prevalence, y_test.shape)
    fitted_label = model_label(args)
    fitted_scores = fit_model(x_train, y_train, x_test, args)
    model_scores = {
        "train_prevalence": prevalence_scores,
        fitted_label: fitted_scores,
    }

    all_metrics = pd.concat(
        [
            evaluate(y_test, prevalence_scores, species, "train_prevalence"),
            evaluate(y_test, fitted_scores, species, fitted_label),
        ],
        ignore_index=True,
    )
    summaries = {
        "train_prevalence": summarize_metrics(
            all_metrics[all_metrics["model"] == "train_prevalence"],
            y_test,
            prevalence_scores,
        ),
        fitted_label: summarize_metrics(
            all_metrics[all_metrics["model"] == fitted_label],
            y_test,
            fitted_scores,
        ),
        "rows": {
            "checklists": int(len(checklists)),
            "train_checklists": int(train_mask.sum()),
            "test_checklists": int(test_mask.sum()),
            "detections_for_top_species": int(labels.sum()),
        },
        "features": list(features.columns),
        "split": split_info,
    }

    checklists_test = checklists.loc[test_mask].copy()
    calibration = build_calibration_table(
        checklists_test,
        y_test,
        model_scores,
        args.calibration_bins,
    )
    summaries["calibration"] = summarize_calibration(calibration)

    split_suffix = "" if args.split == "temporal" else f"_{args.split}"
    model_suffix = model_file_suffix(args)
    metric_path = (
        output_dir
        / f"top{args.top_species}_{args.feature_set}{model_suffix}{split_suffix}_metrics.csv"
    )
    summary_path = (
        output_dir
        / f"top{args.top_species}_{args.feature_set}{model_suffix}{split_suffix}_summary.json"
    )
    calibration_path = (
        output_dir
        / f"top{args.top_species}_{args.feature_set}{model_suffix}{split_suffix}_calibration.csv"
    )
    all_metrics.to_csv(metric_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("\nSummary:")
    print(f"split: {split_info['split']}")
    if split_info["split"] == "spatial-stratified":
        print(
            "spatial split: "
            f"{split_info['test_blocks']} of {split_info['blocks_total']} blocks, "
            f"test fraction={split_info['test_fraction_actual']:.3f}, "
            "balance error="
            f"{split_info['mean_absolute_standardized_balance_error']:.4f}"
        )
    for model_name, summary in summaries.items():
        if isinstance(summary, dict) and "macro_auroc" in summary:
            print(
                f"{model_name}: macro AUROC={summary['macro_auroc']:.4f}, "
                f"macro AUPRC={summary['macro_auprc']:.4f}, "
                f"micro AUROC={summary['micro_auroc']:.4f}, "
                f"micro AUPRC={summary['micro_auprc']:.4f}"
            )
    if summaries["calibration"]:
        print("Calibration:")
        for model_name, summary in summaries["calibration"].items():
            print(
                f"{model_name}: ECE={summary['expected_calibration_error']:.4f}, "
                f"max bin error={summary['max_probability_bin_error']:.4f}"
            )
    print(f"Wrote {metric_path}")
    print(f"Wrote {calibration_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
