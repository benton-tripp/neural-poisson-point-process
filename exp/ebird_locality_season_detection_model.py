"""
Fit a first two-component checklist detection bridge.

This script starts from the locality-season replication dataset and uses it in
two stages:

1. Fit an aggregate locality-season/species availability-style model from
   habitat and biological season features.
2. Broadcast those locality-season/species predictions back to individual
   complete checklists, then fit checklist-level detection models that test
   whether effort and timing explain detection around the availability score.

This is not a full latent occupancy model. It is the first explicit bridge from
the aggregate locality-season baseline toward

    availability(locality, season, species) + detection(checklist, effort, species)

Run from the project root:

    python exp/ebird_locality_season_detection_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --availability-epochs 30 --detection-epochs 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from build_ebird_locality_season_dataset import add_season_columns
from ebird_joint_tabular_baseline import auc_roc, average_precision
from ebird_locality_season_baseline import (
    DEFAULT_FOCUS_SPECIES,
    build_feature_frame,
    fit_model as fit_aggregate_model,
    load_triplets,
    make_split as make_aggregate_split,
    standardize,
    train_species_logits,
)


DEFAULT_DATASET_DIR = "data/ebird/locality_season_top100"
DEFAULT_PROCESSED_DIR = "data/ebird/processed_nc_2020_2023"
DEFAULT_OUTPUT_DIR_NAME = "detection_models"
SEED = 19
EPS = 1e-6

CHECKLIST_COLUMNS = [
    "sampling_event_identifier",
    "locality_id",
    "locality_type",
    "county",
    "county_code",
    "year",
    "month",
    "day_of_year",
    "day_of_week",
    "time_observations_started",
    "protocol_code",
    "duration_minutes",
    "effort_distance_km",
    "number_observers",
]

LOCALITY_SEASON_CONTEXT_COLUMNS = [
    "canopy_median",
    "elevation_median",
    "distance_to_waterbody_m_median",
    "distance_to_coastline_m_median",
]

RESPONSE_COVARIATE_COLUMNS = [
    ("canopy_median", "Canopy median"),
    ("elevation_median", "Elevation median"),
    ("distance_to_waterbody_m_median", "Distance to waterbody median"),
    ("distance_to_coastline_m_median", "Distance to coastline median"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit checklist-level detection models using locality-season availability."
    )
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help=f"Locality-season dataset directory. Defaults to {DEFAULT_DATASET_DIR}.",
    )
    parser.add_argument(
        "--processed-dir",
        default=DEFAULT_PROCESSED_DIR,
        help=f"Processed eBird directory. Defaults to {DEFAULT_PROCESSED_DIR}.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to dataset-dir/detection_models.",
    )
    parser.add_argument(
        "--run-name",
        default="two_component_checklist_detection",
        help="Filename prefix for outputs. Defaults to two_component_checklist_detection.",
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
        help="Add x/y coordinates to the aggregate availability feature set.",
    )
    parser.add_argument(
        "--availability-epochs",
        type=int,
        default=30,
        help="Training epochs for the aggregate availability model. Defaults to 30.",
    )
    parser.add_argument(
        "--detection-epochs",
        type=int,
        default=30,
        help="Training epochs for checklist-level effort/two-component models. Defaults to 30.",
    )
    parser.add_argument(
        "--aggregate-batch-size",
        type=int,
        default=65536,
        help="Batch size for aggregate availability training. Defaults to 65,536.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Checklist batch size for detection training. Defaults to 4,096.",
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
        "--two-component-residual-l2",
        type=float,
        default=0.0,
        help=(
            "L2 penalty on the two-component species intercepts and "
            "species-specific effort weights, shrinking the checklist "
            "correction toward zero. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--two-component-availability-weight-l2",
        type=float,
        default=0.0,
        help=(
            "L2 penalty shrinking two-component species availability weights "
            "toward one. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--two-component-effort-mode",
        choices=("species", "shared", "partial"),
        default="species",
        help=(
            "Effort-response structure for the two-component model. "
            "'species' fits independent species coefficients, 'shared' fits "
            "one response for all species, and 'partial' fits a shared response "
            "plus zero-mean species deviations. Defaults to species."
        ),
    )
    parser.add_argument(
        "--max-checklists",
        type=int,
        default=None,
        help="Optional retained-checklist limit for smoke tests. Defaults to all.",
    )
    parser.add_argument(
        "--max-aggregate-rows",
        type=int,
        default=None,
        help="Optional aggregate-row limit for smoke tests. Defaults to all.",
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of calibration bins. Defaults to 10.",
    )
    parser.add_argument(
        "--response-bins",
        type=int,
        default=8,
        help=(
            "Number of quantile bins for focus-species environmental response "
            "diagnostics. Defaults to 8."
        ),
    )
    parser.add_argument(
        "--response-min-checklists",
        type=int,
        default=500,
        help=(
            "Minimum held-out checklists required for focus-species response "
            "or month diagnostic rows. Defaults to 500."
        ),
    )
    parser.add_argument(
        "--stratum-min-checklists",
        type=int,
        default=1000,
        help=(
            "Minimum held-out checklists required for a stratum diagnostic row. "
            "Defaults to 1,000."
        ),
    )
    parser.add_argument(
        "--cross-stratum-min-checklists",
        type=int,
        default=500,
        help=(
            "Minimum held-out checklists required for cross-stratum calibration "
            "diagnostic rows, such as county-season. Defaults to 500."
        ),
    )
    parser.add_argument(
        "--focus-species",
        nargs="*",
        default=DEFAULT_FOCUS_SPECIES,
        help="Common names included in the focus species season output.",
    )
    return parser.parse_args()


class ChecklistDetectionModel(nn.Module):
    def __init__(
        self,
        species_count: int,
        effort_feature_count: int,
        initial_species_logits: np.ndarray,
        use_availability: bool,
        use_effort: bool,
        effort_mode: str = "species",
    ) -> None:
        super().__init__()
        self.use_availability = use_availability
        self.use_effort = use_effort
        self.effort_mode = effort_mode
        self.species_bias = nn.Parameter(
            torch.as_tensor(initial_species_logits.astype(np.float32))
        )
        if use_availability:
            self.availability_weight = nn.Parameter(torch.ones(species_count))
        else:
            self.register_parameter("availability_weight", None)
        if use_effort and effort_mode == "species":
            self.effort_weights = nn.Parameter(
                torch.zeros(effort_feature_count, species_count)
            )
            self.register_parameter("shared_effort_weights", None)
            self.register_parameter("effort_weight_deviations", None)
        elif use_effort and effort_mode == "shared":
            self.register_parameter("effort_weights", None)
            self.shared_effort_weights = nn.Parameter(
                torch.zeros(effort_feature_count)
            )
            self.register_parameter("effort_weight_deviations", None)
        elif use_effort and effort_mode == "partial":
            self.register_parameter("effort_weights", None)
            self.shared_effort_weights = nn.Parameter(
                torch.zeros(effort_feature_count)
            )
            self.effort_weight_deviations = nn.Parameter(
                torch.zeros(effort_feature_count, species_count)
            )
        elif use_effort:
            raise ValueError(f"Unsupported effort mode: {effort_mode}")
        else:
            self.register_parameter("effort_weights", None)
            self.register_parameter("shared_effort_weights", None)
            self.register_parameter("effort_weight_deviations", None)

    def centered_effort_deviations(self) -> torch.Tensor | None:
        if self.effort_weight_deviations is None:
            return None
        return self.effort_weight_deviations - self.effort_weight_deviations.mean(
            dim=1, keepdim=True
        )

    def effective_effort_weights(self) -> torch.Tensor | None:
        if not self.use_effort:
            return None
        if self.effort_mode == "species":
            return self.effort_weights
        shared = self.shared_effort_weights.unsqueeze(1)
        if self.effort_mode == "shared":
            return shared
        return shared + self.centered_effort_deviations()

    def forward(
        self,
        effort_features: torch.Tensor,
        availability_logits: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.species_bias.unsqueeze(0).expand(
            effort_features.shape[0], -1
        )
        if self.use_availability:
            logits = logits + availability_logits * self.availability_weight.unsqueeze(0)
        if self.use_effort:
            effort_weights = self.effective_effort_weights()
            effort_logits = effort_features @ effort_weights
            if effort_logits.shape[1] == 1:
                effort_logits = effort_logits.expand(-1, logits.shape[1])
            logits = logits + effort_logits
        return logits


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, EPS, 1.0 - EPS)
    return np.log(values / (1.0 - values))


def load_metadata(dataset_dir: Path) -> dict:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def aggregate_availability_scores(
    dataset_dir: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    frame = load_triplets(dataset_dir, args.include_inadequate, args.max_aggregate_rows)
    train_mask, _, _ = make_aggregate_split(frame, args.test_season_year)
    features = build_feature_frame(frame, "availability", args.include_coordinates)
    scaled, feature_metadata = standardize(features, train_mask)
    species_index = frame["species_index"].to_numpy(dtype=np.int64)
    n_checklists = frame["n_checklists"].to_numpy(dtype=np.float32)
    n_detections = frame["n_detections"].to_numpy(dtype=np.float32)
    target_rate = np.divide(
        n_detections,
        n_checklists,
        out=np.zeros_like(n_detections),
        where=n_checklists > 0,
    ).astype(np.float32)
    initial_species_logits = train_species_logits(
        species_index, n_detections, n_checklists, train_mask
    )
    availability_args = argparse.Namespace(
        epochs=args.availability_epochs,
        batch_size=args.aggregate_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scores = fit_aggregate_model(
        scaled,
        species_index,
        target_rate,
        n_checklists,
        train_mask,
        availability_args,
        "aggregate_availability",
        initial_species_logits,
    )
    metadata = {
        "feature_metadata": feature_metadata,
        "rows": int(len(frame)),
        "train_rows": int(train_mask.sum()),
    }
    return frame, scores.astype(np.float32), metadata


def parse_start_hour(values: pd.Series) -> pd.Series:
    text = values.astype("string")
    parsed = pd.to_datetime(text, format="%H:%M:%S", errors="coerce")
    if parsed.isna().all():
        parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    return parsed.dt.hour + parsed.dt.minute / 60.0


def load_and_assign_checklists(
    processed_dir: Path,
    dataset_dir: Path,
    metadata: dict,
    include_inadequate: bool,
    max_checklists: int | None,
) -> gpd.GeoDataFrame:
    checklists = gpd.read_parquet(
        processed_dir / "checklists.geoparquet",
        columns=CHECKLIST_COLUMNS + ["geometry"],
    )
    missing = [col for col in CHECKLIST_COLUMNS if col not in checklists.columns]
    if missing:
        raise ValueError(f"Missing checklist columns: {', '.join(missing)}")

    before = len(checklists)
    checklists = checklists.loc[checklists["locality_id"].notna()].copy()
    included_types = metadata.get("included_locality_types", ["H", "P"])
    if included_types != "all":
        checklists = checklists.loc[checklists["locality_type"].isin(included_types)].copy()
    checklists = add_season_columns(
        checklists, metadata.get("season_scheme", "biological-nc")
    )

    locality_seasons = pd.read_parquet(
        dataset_dir / "locality_seasons.parquet",
        columns=[
            "locality_season_id",
            "locality_id",
            "season_year",
            "season_name",
            "n_checklists",
            "n_dates",
            "unique_observers",
            "duration_bin_count",
            "protocol_count",
            *LOCALITY_SEASON_CONTEXT_COLUMNS,
            "adequate_sampling",
            "eligible_for_species_table",
        ],
    )
    checklists = checklists.merge(
        locality_seasons,
        on=["locality_id", "season_year", "season_name"],
        how="inner",
        validate="many_to_one",
    )
    checklists = checklists.loc[checklists["eligible_for_species_table"]].copy()
    if not include_inadequate:
        checklists = checklists.loc[checklists["adequate_sampling"]].copy()
    if max_checklists is not None:
        checklists = checklists.head(max_checklists).copy()
    checklists["start_hour"] = parse_start_hour(
        checklists["time_observations_started"]
    ).fillna(8.0)
    print(
        f"Retained {len(checklists):,} of {before:,} checklists after locality-season filters."
    )
    return checklists.reset_index(drop=True)


def load_species(dataset_dir: Path) -> pd.DataFrame:
    species = pd.read_csv(dataset_dir / "species.csv")
    if "species_index" not in species.columns:
        species.insert(0, "species_index", np.arange(len(species), dtype=np.int16))
    return species.sort_values("species_index").reset_index(drop=True)


def build_labels(
    checklists: pd.DataFrame,
    detections: pd.DataFrame,
    species: pd.DataFrame,
) -> np.ndarray:
    checklist_index = pd.Series(
        np.arange(len(checklists), dtype=np.int64),
        index=checklists["sampling_event_identifier"],
    )
    species_index = pd.Series(
        species["species_index"].to_numpy(dtype=np.int64),
        index=species["species_key"],
    )
    rows = detections["sampling_event_identifier"].map(checklist_index).to_numpy()
    cols = detections["species_key"].map(species_index).to_numpy()
    valid = (~pd.isna(rows)) & (~pd.isna(cols))
    labels = np.zeros((len(checklists), len(species)), dtype=np.float32)
    labels[rows[valid].astype(np.int64), cols[valid].astype(np.int64)] = 1.0
    return labels


def load_labels(
    processed_dir: Path,
    checklists: pd.DataFrame,
    species: pd.DataFrame,
) -> np.ndarray:
    detections = pd.read_parquet(
        processed_dir / "detections.parquet",
        columns=["sampling_event_identifier", "species_key"],
    )
    detections = detections.loc[
        detections["species_key"].isin(species["species_key"])
        & detections["sampling_event_identifier"].isin(
            checklists["sampling_event_identifier"]
        )
    ]
    detections = detections.drop_duplicates(
        ["sampling_event_identifier", "species_key"]
    )
    return build_labels(checklists, detections, species)


def availability_matrix(
    checklists: pd.DataFrame,
    aggregate_frame: pd.DataFrame,
    aggregate_scores: np.ndarray,
    species_count: int,
) -> np.ndarray:
    work = aggregate_frame[["locality_season_id", "species_index"]].copy()
    work["availability_score"] = aggregate_scores
    pivot = work.pivot(
        index="locality_season_id",
        columns="species_index",
        values="availability_score",
    )
    pivot = pivot.reindex(columns=np.arange(species_count), fill_value=np.nan)
    matrix = pivot.reindex(checklists["locality_season_id"]).to_numpy(dtype=np.float32)
    missing = np.isnan(matrix).sum()
    if missing:
        raise ValueError(
            f"Missing {missing:,} checklist/species availability scores. "
            "Check dataset filtering and max-aggregate-rows."
        )
    return matrix


def add_cyclic(values: pd.Series, period: float, prefix: str) -> pd.DataFrame:
    array = values.astype(float).to_numpy()
    radians = 2.0 * np.pi * array / period
    return pd.DataFrame(
        {
            f"{prefix}_sin": np.sin(radians),
            f"{prefix}_cos": np.cos(radians),
        },
        index=values.index,
    )


def build_effort_features(checklists: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=checklists.index)
    features["duration_log1p"] = np.log1p(
        pd.to_numeric(checklists["duration_minutes"], errors="coerce").fillna(0.0)
    )
    features["effort_distance_log1p"] = np.log1p(
        pd.to_numeric(checklists["effort_distance_km"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0)
    )
    features["number_observers_log1p"] = np.log1p(
        pd.to_numeric(checklists["number_observers"], errors="coerce").fillna(1.0)
    )
    features["is_stationary"] = checklists["protocol_code"].eq("P21").astype(float)
    features["is_traveling"] = checklists["protocol_code"].eq("P22").astype(float)
    features = pd.concat(
        [
            features,
            add_cyclic(checklists["day_of_year"], 366.0, "day_of_year"),
            add_cyclic(checklists["day_of_week"], 7.0, "day_of_week"),
            add_cyclic(checklists["start_hour"], 24.0, "start_hour"),
        ],
        axis=1,
    )
    return features.astype(np.float32)


def standardize_checklist_features(
    features: pd.DataFrame,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, dict]:
    values = features.to_numpy(dtype=np.float32)
    mean = values[train_mask].mean(axis=0, keepdims=True)
    std = values[train_mask].std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    scaled = (values - mean) / std
    return scaled.astype(np.float32), {
        "feature_names": list(features.columns),
        "feature_mean": mean.ravel().astype(float).tolist(),
        "feature_std": std.ravel().astype(float).tolist(),
    }


def species_detection_logits(labels: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    positives = labels[train_mask].sum(axis=0)
    trials = train_mask.sum()
    rates = (positives + 1.0) / (trials + 2.0)
    return logit(rates).astype(np.float32)


def train_detection_model(
    effort_features: np.ndarray,
    availability_logits: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    args: argparse.Namespace,
    model_name: str,
    use_availability: bool,
    use_effort: bool,
    residual_l2: float = 0.0,
    availability_weight_l2: float = 0.0,
    effort_mode: str = "species",
) -> tuple[np.ndarray, dict]:
    torch.manual_seed(SEED)
    if use_availability:
        # Availability logits already include species-specific baseline rates from
        # the aggregate model. Start the checklist-level component as an offset
        # around that score rather than adding a second prevalence intercept.
        initial_logits = np.zeros(labels.shape[1], dtype=np.float32)
    else:
        initial_logits = species_detection_logits(labels, train_mask)
    model = ChecklistDetectionModel(
        labels.shape[1],
        effort_features.shape[1],
        initial_logits,
        use_availability=use_availability,
        use_effort=use_effort,
        effort_mode=effort_mode,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    dataset = TensorDataset(
        torch.from_numpy(effort_features[train_mask].astype(np.float32)),
        torch.from_numpy(availability_logits[train_mask].astype(np.float32)),
        torch.from_numpy(labels[train_mask].astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(1, args.detection_epochs + 1):
        model.train()
        bce_losses = []
        objective_losses = []
        for effort_batch, availability_batch, label_batch in loader:
            optimizer.zero_grad()
            logits = model(effort_batch, availability_batch)
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, label_batch
            )
            objective = bce
            if residual_l2 > 0.0:
                residual_penalty = model.species_bias.square().mean()
                if model.effort_weights is not None:
                    residual_penalty = (
                        residual_penalty + model.effort_weights.square().mean()
                    )
                if model.effort_weight_deviations is not None:
                    residual_penalty = (
                        residual_penalty
                        + model.centered_effort_deviations().square().mean()
                    )
                objective = objective + residual_l2 * residual_penalty
            if availability_weight_l2 > 0.0 and model.availability_weight is not None:
                availability_penalty = (
                    model.availability_weight - 1.0
                ).square().mean()
                objective = objective + availability_weight_l2 * availability_penalty
            objective.backward()
            optimizer.step()
            bce_losses.append(float(bce.detach()))
            objective_losses.append(float(objective.detach()))
        if epoch == 1 or epoch == args.detection_epochs or epoch % 10 == 0:
            message = (
                f"{model_name} epoch {epoch:>4}: "
                f"train BCE={np.mean(bce_losses):.5f}"
            )
            if residual_l2 > 0.0 or availability_weight_l2 > 0.0:
                message += f", objective={np.mean(objective_losses):.5f}"
            print(message)

    effective_effort_weights = model.effective_effort_weights()
    centered_effort_deviations = model.centered_effort_deviations()
    parameter_summary = {
        "species_bias_rms": float(
            torch.sqrt(model.species_bias.detach().square().mean())
        ),
        "effort_weight_rms": (
            float(torch.sqrt(effective_effort_weights.detach().square().mean()))
            if effective_effort_weights is not None
            else None
        ),
        "shared_effort_weight_rms": (
            float(
                torch.sqrt(
                    model.shared_effort_weights.detach().square().mean()
                )
            )
            if model.shared_effort_weights is not None
            else None
        ),
        "effort_weight_deviation_rms": (
            float(
                torch.sqrt(
                    centered_effort_deviations.detach().square().mean()
                )
            )
            if centered_effort_deviations is not None
            else None
        ),
        "effort_mode": model.effort_mode,
        "availability_weight_mean": (
            float(model.availability_weight.detach().mean())
            if model.availability_weight is not None
            else None
        ),
        "availability_weight_deviation_rms": (
            float(
                torch.sqrt(
                    (model.availability_weight.detach() - 1.0).square().mean()
                )
            )
            if model.availability_weight is not None
            else None
        ),
        "residual_l2": float(residual_l2),
        "availability_weight_l2": float(availability_weight_l2),
    }
    return (
        predict_detection_model(
            model, effort_features, availability_logits, args.batch_size
        ),
        parameter_summary,
    )


def predict_detection_model(
    model: ChecklistDetectionModel,
    effort_features: np.ndarray,
    availability_logits: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    scores = np.empty(availability_logits.shape, dtype=np.float32)
    dataset = TensorDataset(
        torch.from_numpy(effort_features.astype(np.float32)),
        torch.from_numpy(availability_logits.astype(np.float32)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    offset = 0
    with torch.no_grad():
        for effort_batch, availability_batch in loader:
            logits = model(effort_batch, availability_batch)
            batch_scores = torch.sigmoid(logits).numpy()
            scores[offset : offset + len(batch_scores)] = batch_scores
            offset += len(batch_scores)
    return scores


def train_prevalence_scores(labels: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    rates = sigmoid(species_detection_logits(labels, train_mask))
    return np.broadcast_to(rates.reshape(1, -1), labels.shape).astype(np.float32)


def calibration_summary(
    y_true: np.ndarray,
    scores: np.ndarray,
    bins: int,
) -> tuple[float, float]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(scores, edges, right=True) - 1, 0, bins - 1)
    total = len(scores)
    ece = 0.0
    max_error = 0.0
    for idx in range(bins):
        mask = bin_ids == idx
        if not mask.any():
            continue
        predicted = float(scores[mask].mean())
        observed = float(y_true[mask].mean())
        error = abs(predicted - observed)
        ece += error * float(mask.sum()) / total
        max_error = max(max_error, error)
    return float(ece), float(max_error)


def calibration_bins_frame(
    y_true: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    bins: int,
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(scores, edges, right=True) - 1, 0, bins - 1)
    rows = []
    for idx in range(bins):
        mask = bin_ids == idx
        if not mask.any():
            continue
        predicted = float(scores[mask].mean())
        observed = float(y_true[mask].mean())
        rows.append(
            {
                "model": model_name,
                "bin": idx,
                "bin_lower": float(edges[idx]),
                "bin_upper": float(edges[idx + 1]),
                "pairs": int(mask.sum()),
                "mean_predicted": predicted,
                "observed_rate": observed,
                "calibration_error": abs(predicted - observed),
            }
        )
    return pd.DataFrame(rows)


def calibration_bins_rows(
    y_true: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    bins: int,
    extra: dict,
) -> list[dict]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(scores, edges, right=True) - 1, 0, bins - 1)
    rows = []
    for idx in range(bins):
        mask = bin_ids == idx
        if not mask.any():
            continue
        predicted = float(scores[mask].mean())
        observed = float(y_true[mask].mean())
        row = {
            **extra,
            "model": model_name,
            "bin": idx,
            "bin_lower": float(edges[idx]),
            "bin_upper": float(edges[idx + 1]),
            "pairs": int(mask.sum()),
            "detections": int(y_true[mask].sum()),
            "mean_predicted": predicted,
            "observed_rate": observed,
            "calibration_error": abs(predicted - observed),
        }
        rows.append(row)
    return rows


def binary_cross_entropy(y_true: np.ndarray, scores: np.ndarray) -> float:
    scores = np.clip(scores, 1e-7, 1.0 - 1e-7)
    return float(
        -(y_true * np.log(scores) + (1.0 - y_true) * np.log1p(-scores)).mean()
    )


def summarize_overall(
    labels: np.ndarray,
    test_mask: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    calibration_bins: int,
) -> dict:
    y = labels[test_mask].ravel()
    p = scores[test_mask].ravel()
    ece, max_error = calibration_summary(y, p, calibration_bins)
    return {
        "model": model_name,
        "checklists": int(test_mask.sum()),
        "pairs": int(len(y)),
        "detections": int(y.sum()),
        "observed_detection_rate": float(y.mean()),
        "mean_predicted_detection_rate": float(p.mean()),
        "calibration_error": abs(float(p.mean()) - float(y.mean())),
        "bce": binary_cross_entropy(y, p),
        "micro_auroc": auc_roc(y, p),
        "micro_auprc": average_precision(y, p),
        "ece": ece,
        "max_bin_error": max_error,
    }


def summarize_by_species(
    labels: np.ndarray,
    test_mask: np.ndarray,
    scores: np.ndarray,
    species: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    y_test = labels[test_mask]
    p_test = scores[test_mask]
    for idx, row in species.iterrows():
        y = y_test[:, idx]
        p = p_test[:, idx]
        rows.append(
            {
                "model": model_name,
                "species_index": int(row["species_index"]),
                "species_key": row["species_key"],
                "common_name": row["common_name"],
                "scientific_name": row["scientific_name"],
                "checklists": int(len(y)),
                "detections": int(y.sum()),
                "observed_detection_rate": float(y.mean()),
                "mean_predicted_detection_rate": float(p.mean()),
                "calibration_error": abs(float(p.mean()) - float(y.mean())),
                "auroc": auc_roc(y, p),
                "auprc": average_precision(y, p),
            }
        )
    return pd.DataFrame(rows)


def species_metric_means(species_metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        species_metrics.groupby("model", observed=True)[
            ["auroc", "auprc", "calibration_error"]
        ]
        .mean()
        .reset_index()
        .sort_values("model")
    )


def species_delta_vs_availability(species_metrics: pd.DataFrame) -> pd.DataFrame:
    wide = species_metrics.pivot(
        index=["species_key", "common_name", "scientific_name"],
        columns="model",
        values="auprc",
    ).reset_index()
    if "availability_only" not in wide.columns or "two_component" not in wide.columns:
        return pd.DataFrame()
    wide["delta_two_component_vs_availability"] = (
        wide["two_component"] - wide["availability_only"]
    )
    return wide.sort_values(
        "delta_two_component_vs_availability", ascending=False
    )


def summarize_focus_species_season(
    checklists: pd.DataFrame,
    labels: np.ndarray,
    test_mask: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    species: pd.DataFrame,
    focus_species: list[str],
) -> pd.DataFrame:
    focus_indices = species.loc[
        species["common_name"].isin(focus_species), ["species_index", "common_name"]
    ]
    rows = []
    test_checklists = checklists.loc[test_mask, ["season_name"]].reset_index(drop=True)
    for _, species_row in focus_indices.iterrows():
        idx = int(species_row["species_index"])
        common_name = species_row["common_name"]
        work = test_checklists.copy()
        work["detected"] = labels[test_mask, idx]
        for model_name, scores in scores_by_model.items():
            work[f"{model_name}_score"] = scores[test_mask, idx]
        for season_name, group in work.groupby("season_name", observed=True):
            y = group["detected"].to_numpy(dtype=float)
            row = {
                "common_name": common_name,
                "season_name": season_name,
                "checklists": int(len(group)),
                "detections": int(y.sum()),
                "observed_detection_rate": float(y.mean()),
            }
            for model_name in scores_by_model:
                p = group[f"{model_name}_score"].to_numpy(dtype=float)
                row[f"{model_name}_predicted_detection_rate"] = float(p.mean())
                row[f"{model_name}_calibration_error"] = abs(
                    float(p.mean()) - row["observed_detection_rate"]
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["common_name", "season_name"])


def effort_distance_stratum(values: pd.Series) -> pd.Series:
    distance = pd.to_numeric(values, errors="coerce")
    output = pd.Series("missing", index=values.index, dtype="object")
    output.loc[distance.eq(0.0)] = "0"
    output.loc[distance.gt(0.0) & distance.le(0.5)] = "(0,0.5]"
    output.loc[distance.gt(0.5) & distance.le(2.0)] = "(0.5,2]"
    output.loc[distance.gt(2.0) & distance.le(5.0)] = "(2,5]"
    output.loc[distance.gt(5.0)] = "5+"
    return output


def observer_stratum(values: pd.Series) -> pd.Series:
    observers = pd.to_numeric(values, errors="coerce")
    output = pd.Series("missing", index=values.index, dtype="object")
    output.loc[observers.eq(1.0)] = "1"
    output.loc[observers.eq(2.0)] = "2"
    output.loc[observers.ge(3.0)] = "3+"
    return output


def start_hour_stratum(values: pd.Series) -> pd.Series:
    hour = pd.to_numeric(values, errors="coerce")
    output = pd.Series("missing", index=values.index, dtype="object")
    output.loc[hour.lt(8.0)] = "before_8"
    output.loc[hour.ge(8.0) & hour.lt(11.0)] = "8_to_11"
    output.loc[hour.ge(11.0) & hour.lt(14.0)] = "11_to_14"
    output.loc[hour.ge(14.0) & hour.lt(18.0)] = "14_to_18"
    output.loc[hour.ge(18.0)] = "18_plus"
    return output


def add_strata_columns(checklists: pd.DataFrame) -> pd.DataFrame:
    strata = pd.DataFrame(index=checklists.index)
    protocol_labels = {
        "P21": "Stationary",
        "P22": "Traveling",
    }
    strata["protocol"] = (
        checklists["protocol_code"]
        .map(protocol_labels)
        .fillna(checklists["protocol_code"].astype("string"))
        .fillna("missing")
        .astype(str)
    )
    strata["duration_minutes"] = pd.cut(
        pd.to_numeric(checklists["duration_minutes"], errors="coerce"),
        bins=[-np.inf, 10.0, 30.0, 60.0, 120.0, np.inf],
        labels=["1-10", "11-30", "31-60", "61-120", "121+"],
    ).astype("string").fillna("missing")
    strata["effort_distance_km"] = effort_distance_stratum(
        checklists["effort_distance_km"]
    )
    strata["number_observers"] = observer_stratum(checklists["number_observers"])
    strata["start_hour"] = start_hour_stratum(checklists["start_hour"])
    strata["season_name"] = checklists["season_name"].astype("string").fillna("missing")
    strata["locality_type"] = (
        checklists["locality_type"].astype("string").fillna("missing")
    )
    strata["county"] = checklists["county"].astype("string").fillna("missing")
    strata["locality_season_checklists"] = pd.cut(
        pd.to_numeric(checklists["n_checklists"], errors="coerce"),
        bins=[0.0, 2.0, 5.0, 10.0, 25.0, 50.0, np.inf],
        labels=["1-2", "3-5", "6-10", "11-25", "26-50", "51+"],
        include_lowest=True,
    ).astype("string").fillna("missing")
    strata["locality_season_dates"] = pd.cut(
        pd.to_numeric(checklists["n_dates"], errors="coerce"),
        bins=[0.0, 2.0, 5.0, 10.0, 25.0, np.inf],
        labels=["1-2", "3-5", "6-10", "11-25", "26+"],
        include_lowest=True,
    ).astype("string").fillna("missing")
    strata["locality_season_effort_bins"] = (
        pd.to_numeric(checklists["duration_bin_count"], errors="coerce")
        .fillna(-1)
        .astype(int)
        .astype(str)
    )
    return strata


def summarize_by_strata(
    checklists: pd.DataFrame,
    labels: np.ndarray,
    test_mask: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    calibration_bins: int,
    min_checklists: int,
) -> pd.DataFrame:
    test_indices = np.flatnonzero(test_mask)
    strata = add_strata_columns(checklists.loc[test_mask]).reset_index(drop=True)
    strata["row_index"] = test_indices
    stratum_columns = [
        "protocol",
        "duration_minutes",
        "effort_distance_km",
        "number_observers",
        "start_hour",
        "season_name",
        "locality_type",
        "county",
        "locality_season_checklists",
        "locality_season_dates",
        "locality_season_effort_bins",
    ]
    rows = []
    for stratum_type in stratum_columns:
        grouped = strata.groupby(stratum_type, dropna=False, sort=True)
        for stratum, group in grouped:
            if len(group) < min_checklists:
                continue
            row_indices = group["row_index"].to_numpy(dtype=np.int64)
            y = labels[row_indices].ravel()
            observed = float(y.mean())
            detections = int(y.sum())
            for model_name, scores in scores_by_model.items():
                p = scores[row_indices].ravel()
                ece, max_error = calibration_summary(y, p, calibration_bins)
                rows.append(
                    {
                        "stratum_type": stratum_type,
                        "stratum": str(stratum),
                        "model": model_name,
                        "checklists": int(len(group)),
                        "pairs": int(len(y)),
                        "detections": detections,
                        "observed_detection_rate": observed,
                        "mean_predicted_detection_rate": float(p.mean()),
                        "calibration_error": abs(float(p.mean()) - observed),
                        "bce": binary_cross_entropy(y, p),
                        "micro_auroc": auc_roc(y, p),
                        "micro_auprc": average_precision(y, p),
                        "ece": ece,
                        "max_bin_error": max_error,
                    }
                )
    return pd.DataFrame(rows)


def stratum_deltas(strata_metrics: pd.DataFrame) -> pd.DataFrame:
    if strata_metrics.empty:
        return pd.DataFrame()
    metric_columns = [
        "bce",
        "micro_auroc",
        "micro_auprc",
        "ece",
        "calibration_error",
        "mean_predicted_detection_rate",
    ]
    base_columns = [
        "stratum_type",
        "stratum",
        "checklists",
        "pairs",
        "detections",
        "observed_detection_rate",
    ]
    rows = []
    for (stratum_type, stratum), group in strata_metrics.groupby(
        ["stratum_type", "stratum"], observed=True, sort=True
    ):
        indexed = group.set_index("model")
        if "two_component" not in indexed.index:
            continue
        for baseline in ["availability_only", "effort_only", "train_prevalence"]:
            if baseline not in indexed.index:
                continue
            row = {
                "stratum_type": stratum_type,
                "stratum": stratum,
                "baseline_model": baseline,
            }
            for col in base_columns[2:]:
                row[col] = indexed.loc["two_component", col]
            for metric in metric_columns:
                row[f"delta_{metric}"] = (
                    indexed.loc["two_component", metric]
                    - indexed.loc[baseline, metric]
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["baseline_model", "delta_micro_auprc"], ascending=[True, False]
    )


def summarize_county_season_diagnostics(
    checklists: pd.DataFrame,
    labels: np.ndarray,
    test_mask: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    calibration_bins: int,
    min_checklists: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_indices = np.flatnonzero(test_mask)
    strata = add_strata_columns(checklists.loc[test_mask]).reset_index(drop=True)
    strata["row_index"] = test_indices
    strata["county_season"] = (
        strata["county"].astype(str) + "|" + strata["season_name"].astype(str)
    )

    metric_rows = []
    calibration_rows = []
    for stratum, group in strata.groupby("county_season", dropna=False, sort=True):
        if len(group) < min_checklists:
            continue
        row_indices = group["row_index"].to_numpy(dtype=np.int64)
        y = labels[row_indices].ravel()
        observed = float(y.mean())
        detections = int(y.sum())
        county, season_name = str(stratum).split("|", 1)
        extra = {
            "stratum_type": "county_season",
            "stratum": str(stratum),
            "county": county,
            "season_name": season_name,
            "checklists": int(len(group)),
        }
        for model_name, scores in scores_by_model.items():
            p = scores[row_indices].ravel()
            ece, max_error = calibration_summary(y, p, calibration_bins)
            metric_rows.append(
                {
                    **extra,
                    "model": model_name,
                    "pairs": int(len(y)),
                    "detections": detections,
                    "observed_detection_rate": observed,
                    "mean_predicted_detection_rate": float(p.mean()),
                    "calibration_error": abs(float(p.mean()) - observed),
                    "bce": binary_cross_entropy(y, p),
                    "micro_auroc": auc_roc(y, p),
                    "micro_auprc": average_precision(y, p),
                    "ece": ece,
                    "max_bin_error": max_error,
                }
            )
            calibration_rows.extend(
                calibration_bins_rows(
                    y,
                    p,
                    model_name,
                    calibration_bins,
                    extra,
                )
            )
    metrics = pd.DataFrame(metric_rows)
    deltas = stratum_deltas(metrics) if not metrics.empty else pd.DataFrame()
    return metrics, deltas, pd.DataFrame(calibration_rows)


def summarize_focus_species_season_calibration(
    checklists: pd.DataFrame,
    labels: np.ndarray,
    test_mask: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    species: pd.DataFrame,
    focus_species: list[str],
    calibration_bins: int,
) -> pd.DataFrame:
    focus_indices = species.loc[
        species["common_name"].isin(focus_species), ["species_index", "common_name"]
    ]
    test_checklists = checklists.loc[test_mask, ["season_name"]].reset_index(drop=True)
    rows = []
    for _, species_row in focus_indices.iterrows():
        idx = int(species_row["species_index"])
        common_name = species_row["common_name"]
        work = test_checklists.copy()
        work["detected"] = labels[test_mask, idx]
        for model_name, scores in scores_by_model.items():
            work[f"{model_name}_score"] = scores[test_mask, idx]
        for season_name, group in work.groupby("season_name", observed=True):
            y = group["detected"].to_numpy(dtype=float)
            extra = {
                "common_name": common_name,
                "species_index": idx,
                "season_name": season_name,
                "checklists": int(len(group)),
            }
            for model_name in scores_by_model:
                p = group[f"{model_name}_score"].to_numpy(dtype=float)
                rows.extend(
                    calibration_bins_rows(
                        y,
                        p,
                        model_name,
                        calibration_bins,
                        extra,
                    )
                )
    return pd.DataFrame(rows)


def summarize_focus_species_month(
    checklists: pd.DataFrame,
    labels: np.ndarray,
    test_mask: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    species: pd.DataFrame,
    focus_species: list[str],
    min_checklists: int,
) -> pd.DataFrame:
    focus_indices = species.loc[
        species["common_name"].isin(focus_species), ["species_index", "common_name"]
    ]
    test_checklists = checklists.loc[test_mask, ["month"]].reset_index(drop=True)
    rows = []
    for _, species_row in focus_indices.iterrows():
        idx = int(species_row["species_index"])
        common_name = species_row["common_name"]
        work = test_checklists.copy()
        work["detected"] = labels[test_mask, idx]
        for model_name, scores in scores_by_model.items():
            work[f"{model_name}_score"] = scores[test_mask, idx]
        for month, group in work.groupby("month", observed=True, sort=True):
            if len(group) < min_checklists:
                continue
            y = group["detected"].to_numpy(dtype=float)
            row = {
                "common_name": common_name,
                "species_index": idx,
                "month": int(month),
                "checklists": int(len(group)),
                "detections": int(y.sum()),
                "observed_detection_rate": float(y.mean()),
            }
            for model_name in scores_by_model:
                p = group[f"{model_name}_score"].to_numpy(dtype=float)
                row[f"{model_name}_predicted_detection_rate"] = float(p.mean())
                row[f"{model_name}_calibration_error"] = abs(
                    float(p.mean()) - row["observed_detection_rate"]
                )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["common_name", "month"])


def response_bins(values: pd.Series, bins: int) -> pd.DataFrame:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.notna()
    valid_values = numeric.loc[valid]
    if valid_values.nunique() < 2:
        return pd.DataFrame()
    q = min(int(bins), int(valid_values.nunique()))
    categorized = pd.qcut(valid_values, q=q, duplicates="drop")
    if categorized.nunique() == 0:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "test_position": np.flatnonzero(valid.to_numpy()),
            "bin_id": categorized.cat.codes.to_numpy(dtype=int),
            "bin_label": categorized.astype(str).to_numpy(),
            "covariate_value": valid_values.to_numpy(dtype=float),
        }
    )


def summarize_focus_species_response(
    checklists: pd.DataFrame,
    labels: np.ndarray,
    test_mask: np.ndarray,
    scores_by_model: dict[str, np.ndarray],
    species: pd.DataFrame,
    focus_species: list[str],
    bins: int,
    min_checklists: int,
) -> pd.DataFrame:
    focus_indices = species.loc[
        species["common_name"].isin(focus_species), ["species_index", "common_name"]
    ]
    test_checklists = checklists.loc[test_mask].reset_index(drop=True)
    rows = []
    for covariate, covariate_label in RESPONSE_COVARIATE_COLUMNS:
        if covariate not in test_checklists.columns:
            continue
        bin_frame = response_bins(test_checklists[covariate], bins)
        if bin_frame.empty:
            continue
        for _, species_row in focus_indices.iterrows():
            idx = int(species_row["species_index"])
            common_name = species_row["common_name"]
            species_labels = labels[test_mask, idx]
            species_scores = {
                model_name: scores[test_mask, idx]
                for model_name, scores in scores_by_model.items()
            }
            for (bin_id, bin_label), group in bin_frame.groupby(
                ["bin_id", "bin_label"], sort=True
            ):
                if len(group) < min_checklists:
                    continue
                positions = group["test_position"].to_numpy(dtype=np.int64)
                y = species_labels[positions].astype(float)
                observed = float(y.mean())
                row = {
                    "common_name": common_name,
                    "species_index": idx,
                    "covariate": covariate,
                    "covariate_label": covariate_label,
                    "bin_id": int(bin_id),
                    "bin_label": str(bin_label),
                    "checklists": int(len(group)),
                    "detections": int(y.sum()),
                    "covariate_min": float(group["covariate_value"].min()),
                    "covariate_mean": float(group["covariate_value"].mean()),
                    "covariate_max": float(group["covariate_value"].max()),
                    "observed_detection_rate": observed,
                }
                for model_name, scores in species_scores.items():
                    p = scores[positions].astype(float)
                    row[f"{model_name}_predicted_detection_rate"] = float(p.mean())
                    row[f"{model_name}_calibration_error"] = abs(
                        float(p.mean()) - observed
                    )
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["common_name", "covariate", "bin_id"])


def write_outputs(
    output_dir: Path,
    run_name: str,
    overall: pd.DataFrame,
    species_metrics: pd.DataFrame,
    species_delta: pd.DataFrame,
    calibration: pd.DataFrame,
    focus: pd.DataFrame,
    strata_metrics: pd.DataFrame,
    strata_delta: pd.DataFrame,
    county_season_metrics: pd.DataFrame,
    county_season_delta: pd.DataFrame,
    county_season_calibration: pd.DataFrame,
    focus_season_calibration: pd.DataFrame,
    focus_month: pd.DataFrame,
    focus_response: pd.DataFrame,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    overall.to_csv(output_dir / f"{run_name}_metrics.csv", index=False)
    species_metrics.to_csv(output_dir / f"{run_name}_species_metrics.csv", index=False)
    species_delta.to_csv(
        output_dir / f"{run_name}_species_delta_vs_availability.csv", index=False
    )
    calibration.to_csv(output_dir / f"{run_name}_calibration.csv", index=False)
    focus.to_csv(output_dir / f"{run_name}_focus_species_season.csv", index=False)
    strata_metrics.to_csv(output_dir / f"{run_name}_strata_metrics.csv", index=False)
    strata_delta.to_csv(output_dir / f"{run_name}_strata_deltas.csv", index=False)
    county_season_metrics.to_csv(
        output_dir / f"{run_name}_county_season_metrics.csv", index=False
    )
    county_season_delta.to_csv(
        output_dir / f"{run_name}_county_season_deltas.csv", index=False
    )
    county_season_calibration.to_csv(
        output_dir / f"{run_name}_county_season_calibration.csv", index=False
    )
    focus_season_calibration.to_csv(
        output_dir / f"{run_name}_focus_species_season_calibration.csv",
        index=False,
    )
    focus_month.to_csv(output_dir / f"{run_name}_focus_species_month.csv", index=False)
    focus_response.to_csv(
        output_dir / f"{run_name}_focus_species_response.csv", index=False
    )
    (output_dir / f"{run_name}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    processed_dir = Path(args.processed_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else dataset_dir / DEFAULT_OUTPUT_DIR_NAME
    )
    metadata = load_metadata(dataset_dir)
    species = load_species(dataset_dir)

    aggregate_frame, availability_scores, availability_metadata = (
        aggregate_availability_scores(dataset_dir, args)
    )
    checklists = load_and_assign_checklists(
        processed_dir,
        dataset_dir,
        metadata,
        args.include_inadequate,
        args.max_checklists,
    )
    labels = load_labels(processed_dir, checklists, species)
    train_mask = checklists["season_year"].to_numpy(dtype=int) < args.test_season_year
    test_mask = checklists["season_year"].to_numpy(dtype=int) == args.test_season_year
    unused_mask = checklists["season_year"].to_numpy(dtype=int) > args.test_season_year
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            "Checklist split failed: "
            f"train={train_mask.sum()}, test={test_mask.sum()}."
        )

    availability = availability_matrix(
        checklists, aggregate_frame, availability_scores, len(species)
    )
    availability_logits = logit(availability).astype(np.float32)
    effort_features, effort_metadata = standardize_checklist_features(
        build_effort_features(checklists), train_mask
    )

    print(
        "Checklist rows: "
        f"train={int(train_mask.sum()):,}, test={int(test_mask.sum()):,}, "
        f"unused={int(unused_mask.sum()):,}"
    )
    print(
        "Checklist-species pairs: "
        f"train={int(train_mask.sum() * len(species)):,}, "
        f"test={int(test_mask.sum() * len(species)):,}"
    )

    scores_by_model: dict[str, np.ndarray] = {
        "train_prevalence": train_prevalence_scores(labels, train_mask),
        "availability_only": availability,
    }
    effort_only_scores, effort_only_parameters = train_detection_model(
        effort_features,
        availability_logits,
        labels,
        train_mask,
        args,
        "effort_only",
        use_availability=False,
        use_effort=True,
    )
    scores_by_model["effort_only"] = effort_only_scores
    two_component_scores, two_component_parameters = train_detection_model(
        effort_features,
        availability_logits,
        labels,
        train_mask,
        args,
        "two_component",
        use_availability=True,
        use_effort=True,
        residual_l2=args.two_component_residual_l2,
        availability_weight_l2=args.two_component_availability_weight_l2,
        effort_mode=args.two_component_effort_mode,
    )
    scores_by_model["two_component"] = two_component_scores

    overall = pd.DataFrame(
        [
            summarize_overall(
                labels,
                test_mask,
                scores,
                model_name,
                args.calibration_bins,
            )
            for model_name, scores in scores_by_model.items()
        ]
    )
    species_metrics = pd.concat(
        [
            summarize_by_species(labels, test_mask, scores, species, model_name)
            for model_name, scores in scores_by_model.items()
        ],
        ignore_index=True,
    )
    species_means = species_metric_means(species_metrics)
    species_delta = species_delta_vs_availability(species_metrics)
    calibration = pd.concat(
        [
            calibration_bins_frame(
                labels[test_mask].ravel(),
                scores[test_mask].ravel(),
                model_name,
                args.calibration_bins,
            )
            for model_name, scores in scores_by_model.items()
        ],
        ignore_index=True,
    )
    focus = summarize_focus_species_season(
        checklists,
        labels,
        test_mask,
        scores_by_model,
        species,
        args.focus_species,
    )
    strata_metrics = summarize_by_strata(
        checklists,
        labels,
        test_mask,
        scores_by_model,
        args.calibration_bins,
        args.stratum_min_checklists,
    )
    strata_delta = stratum_deltas(strata_metrics)
    (
        county_season_metrics,
        county_season_delta,
        county_season_calibration,
    ) = summarize_county_season_diagnostics(
        checklists,
        labels,
        test_mask,
        scores_by_model,
        args.calibration_bins,
        args.cross_stratum_min_checklists,
    )
    focus_season_calibration = summarize_focus_species_season_calibration(
        checklists,
        labels,
        test_mask,
        scores_by_model,
        species,
        args.focus_species,
        args.calibration_bins,
    )
    focus_month = summarize_focus_species_month(
        checklists,
        labels,
        test_mask,
        scores_by_model,
        species,
        args.focus_species,
        args.response_min_checklists,
    )
    focus_response = summarize_focus_species_response(
        checklists,
        labels,
        test_mask,
        scores_by_model,
        species,
        args.focus_species,
        args.response_bins,
        args.response_min_checklists,
    )
    summary = {
        "dataset_dir": str(dataset_dir),
        "processed_dir": str(processed_dir),
        "output_dir": str(output_dir),
        "run_name": args.run_name,
        "test_season_year": int(args.test_season_year),
        "include_inadequate": bool(args.include_inadequate),
        "stratum_min_checklists": int(args.stratum_min_checklists),
        "cross_stratum_min_checklists": int(args.cross_stratum_min_checklists),
        "response_bins": int(args.response_bins),
        "response_min_checklists": int(args.response_min_checklists),
        "two_component_regularization": {
            "residual_l2": float(args.two_component_residual_l2),
            "availability_weight_l2": float(
                args.two_component_availability_weight_l2
            ),
            "effort_mode": args.two_component_effort_mode,
        },
        "detection_parameters": {
            "effort_only": effort_only_parameters,
            "two_component": two_component_parameters,
        },
        "rows": {
            "train_checklists": int(train_mask.sum()),
            "test_checklists": int(test_mask.sum()),
            "unused_checklists": int(unused_mask.sum()),
            "species": int(len(species)),
            "strata_metric_rows": int(len(strata_metrics)),
            "strata_delta_rows": int(len(strata_delta)),
            "county_season_metric_rows": int(len(county_season_metrics)),
            "county_season_delta_rows": int(len(county_season_delta)),
            "county_season_calibration_rows": int(len(county_season_calibration)),
            "focus_season_calibration_rows": int(len(focus_season_calibration)),
            "focus_month_rows": int(len(focus_month)),
            "focus_response_rows": int(len(focus_response)),
        },
        "models": overall.set_index("model").to_dict(orient="index"),
        "species_metric_means": species_means.set_index("model").to_dict(
            orient="index"
        ),
        "availability": availability_metadata,
        "effort_features": effort_metadata,
    }
    write_outputs(
        output_dir,
        args.run_name,
        overall,
        species_metrics,
        species_delta,
        calibration,
        focus,
        strata_metrics,
        strata_delta,
        county_season_metrics,
        county_season_delta,
        county_season_calibration,
        focus_season_calibration,
        focus_month,
        focus_response,
        summary,
    )

    print()
    print("Checklist-level detection metrics:")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print()
    print("Mean species-level metrics:")
    print(species_means.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if not species_delta.empty:
        print()
        print("Largest two-component species AUPRC gains over availability-only:")
        gain_cols = [
            "common_name",
            "availability_only",
            "two_component",
            "delta_two_component_vs_availability",
        ]
        print(
            species_delta.head(10)[gain_cols].to_string(
                index=False, float_format=lambda x: f"{x:.5f}"
            )
        )
        print()
        print("Largest two-component species AUPRC losses vs availability-only:")
        print(
            species_delta.tail(10)
            .sort_values("delta_two_component_vs_availability")
            [gain_cols]
            .to_string(index=False, float_format=lambda x: f"{x:.5f}")
        )
    print()
    print("Focus species/season rates:")
    display_cols = [
        "common_name",
        "season_name",
        "checklists",
        "observed_detection_rate",
        "availability_only_predicted_detection_rate",
        "effort_only_predicted_detection_rate",
        "two_component_predicted_detection_rate",
    ]
    if not focus.empty:
        print(focus[display_cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    if not strata_delta.empty:
        availability_delta = strata_delta.loc[
            strata_delta["baseline_model"].eq("availability_only")
        ]
        if not availability_delta.empty:
            display_strata_cols = [
                "stratum_type",
                "stratum",
                "checklists",
                "observed_detection_rate",
                "delta_micro_auprc",
                "delta_ece",
                "delta_calibration_error",
            ]
            print()
            print("Largest two-component stratum AUPRC gains over availability-only:")
            print(
                availability_delta.head(12)[display_strata_cols].to_string(
                    index=False, float_format=lambda x: f"{x:.5f}"
                )
            )
            print()
            if (availability_delta["delta_micro_auprc"] < 0.0).any():
                print("Largest two-component stratum AUPRC losses vs availability-only:")
            else:
                print("Smallest two-component stratum AUPRC gains vs availability-only:")
            print(
                availability_delta.tail(12)
                .sort_values("delta_micro_auprc")
                [display_strata_cols]
                .to_string(index=False, float_format=lambda x: f"{x:.5f}")
            )
    if not county_season_metrics.empty:
        county_season_two_component = county_season_metrics.loc[
            county_season_metrics["model"].eq("two_component")
        ].sort_values("ece", ascending=False)
        if not county_season_two_component.empty:
            county_season_cols = [
                "county",
                "season_name",
                "checklists",
                "observed_detection_rate",
                "mean_predicted_detection_rate",
                "ece",
                "max_bin_error",
                "micro_auprc",
            ]
            print()
            print("Worst two-component county-season calibration:")
            print(
                county_season_two_component.head(12)[county_season_cols].to_string(
                    index=False, float_format=lambda x: f"{x:.5f}"
                )
            )
    print()
    print(
        "Focus plausibility diagnostic rows: "
        f"month={len(focus_month):,}, response={len(focus_response):,}"
    )
    print()
    print(f"Wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
