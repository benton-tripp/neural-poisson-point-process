"""
Fit a first latent repeated-visit availability/detection model.

This script uses the locality-season replication dataset directly. For each
species j and locality-season group g, it models:

    psi[j, g] = P(species j is available at locality-season g)
    p[j, i]   = P(species j is detected on checklist i | available)

The repeated-visit likelihood is:

    if any detection in group:
        log psi + sum_i log Bernoulli(y_i | p_i)

    if no detections in group:
        log((1 - psi) + psi * product_i(1 - p_i))

This is intentionally simple: no GNN, no spatial residual, and no external
anchors. The goal is to test whether the complete-checklist replication itself
can separate locality-season availability from checklist-level detection.

Run from the project root:

    python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 20 --run-name latent_repeated_visit_e20
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ebird_joint_tabular_baseline import auc_roc, average_precision
from ebird_locality_season_baseline import (
    DEFAULT_FOCUS_SPECIES,
    build_feature_frame,
    standardize,
)
from ebird_locality_season_detection_model import (
    build_effort_features,
    calibration_summary,
    load_and_assign_checklists,
    load_labels,
    load_metadata,
    load_species,
    logit,
    sigmoid,
    species_detection_logits,
    summarize_by_species,
    summarize_focus_species_season,
    summarize_overall,
    standardize_checklist_features,
)


DEFAULT_DATASET_DIR = "data/ebird/locality_season_top100"
DEFAULT_PROCESSED_DIR = "data/ebird/processed_nc_2020_2023"
DEFAULT_OUTPUT_DIR_NAME = "latent_models"
SEED = 19
EPS = 1e-7
WINDOWS_MAX_PATH_LENGTH = 259
OUTPUT_SUFFIXES = [
    "_metrics.csv",
    "_species_metrics.csv",
    "_availability_metrics.csv",
    "_availability_species_metrics.csv",
    "_latent_detection_diagnostics.csv",
    "_focus_species_season.csv",
    "_focus_species_availability_season.csv",
    "_focus_species_group_predictions.csv",
    "_component_support_metrics.csv",
    "_component_species_season_metrics.csv",
    "_component_species_support_metrics.csv",
    "_pair_codetection_support_metrics.csv",
    "_pair_codetection_species_season_metrics.csv",
    "_frailty_species.csv",
    "_summary.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a latent repeated-visit locality-season availability/detection model."
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
        help="Output directory. Defaults to dataset-dir/latent_models.",
    )
    parser.add_argument(
        "--run-name",
        default="latent_repeated_visit",
        help="Filename prefix for outputs. Defaults to latent_repeated_visit.",
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
        help="Add x/y coordinates to availability features.",
    )
    parser.add_argument(
        "--min-group-checklists",
        type=int,
        default=None,
        help="Optional stricter minimum checklist count per locality-season.",
    )
    parser.add_argument(
        "--min-group-dates",
        type=int,
        default=None,
        help="Optional stricter minimum distinct-date count per locality-season.",
    )
    parser.add_argument(
        "--min-group-duration-bins",
        type=int,
        default=None,
        help="Optional stricter minimum duration-bin count per locality-season.",
    )
    parser.add_argument(
        "--min-group-protocols",
        type=int,
        default=None,
        help="Optional stricter minimum protocol count per locality-season.",
    )
    parser.add_argument(
        "--min-group-observers",
        type=int,
        default=None,
        help="Optional stricter minimum unique-observer count per locality-season.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Training epochs. Defaults to 20.",
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
        "--availability-l2",
        type=float,
        default=0.0,
        help="Optional L2 penalty on availability weights. Defaults to 0.",
    )
    parser.add_argument(
        "--detection-l2",
        type=float,
        default=0.0,
        help="Optional L2 penalty on detection weights. Defaults to 0.",
    )
    parser.add_argument(
        "--marginal-rate-l2",
        type=float,
        default=0.0,
        help=(
            "Optional training penalty anchoring the overall prior marginal "
            "detection rate to the observed training detection rate. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--species-marginal-rate-l2",
        type=float,
        default=0.0,
        help=(
            "Optional training penalty anchoring species-wise prior marginal "
            "detection rates to observed training rates. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--species-season-mode",
        choices=["none", "availability", "detection", "both"],
        default="none",
        help=(
            "Optional species-by-season latent offset. 'availability' adds it "
            "to psi logits, 'detection' adds it to conditional detection logits, "
            "'both' adds separate offsets to both components. Defaults to none."
        ),
    )
    parser.add_argument(
        "--species-season-l2",
        type=float,
        default=0.0,
        help="Optional L2 penalty on species-by-season offsets. Defaults to 0.",
    )
    parser.add_argument(
        "--detection-frailty-mode",
        choices=["none", "global", "species", "hierarchical"],
        default="none",
        help=(
            "Optional logistic-normal shared detection frailty for repeated "
            "checklists within each locality-season/species. 'global' learns "
            "one standard deviation shared by all species; 'species' learns "
            "one independently per species; 'hierarchical' learns a shared "
            "global scale plus zero-centered species deviations. Defaults to none."
        ),
    )
    parser.add_argument(
        "--detection-frailty-init",
        type=float,
        default=0.5,
        help="Initial logistic-normal detection frailty standard deviation. Defaults to 0.5.",
    )
    parser.add_argument(
        "--detection-frailty-l2",
        type=float,
        default=0.0,
        help="Optional L2 penalty on detection frailty standard deviations. Defaults to 0.",
    )
    parser.add_argument(
        "--detection-frailty-deviation-l2",
        type=float,
        default=0.0,
        help=(
            "Optional L2 penalty on centered species deviations in hierarchical "
            "frailty mode. Defaults to 0."
        ),
    )
    parser.add_argument(
        "--frailty-quadrature-points",
        type=int,
        default=7,
        help="Gauss-Hermite quadrature points for detection frailty. Defaults to 7.",
    )
    parser.add_argument(
        "--max-groups-per-split",
        type=int,
        default=None,
        help=(
            "Optional train/test group limit for smoke tests. Samples up to this "
            "many locality-seasons from each split. Defaults to all groups."
        ),
    )
    parser.add_argument(
        "--calibration-bins",
        type=int,
        default=10,
        help="Number of calibration bins. Defaults to 10.",
    )
    parser.add_argument(
        "--focus-species",
        nargs="*",
        default=DEFAULT_FOCUS_SPECIES,
        help="Common names included in focus species outputs.",
    )
    return parser.parse_args()


def validate_output_paths(output_dir: Path, run_name: str) -> None:
    """Fail before training when a Windows artifact path exceeds MAX_PATH."""
    if os.name != "nt":
        return
    resolved_dir = output_dir.resolve()
    candidates = [
        resolved_dir / f"{run_name}{suffix}" for suffix in OUTPUT_SUFFIXES
    ]
    longest = max(candidates, key=lambda path: len(str(path)))
    if len(str(longest)) <= WINDOWS_MAX_PATH_LENGTH:
        return

    longest_suffix = max(OUTPUT_SUFFIXES, key=len)
    max_run_name_length = max(
        WINDOWS_MAX_PATH_LENGTH
        - len(str(resolved_dir))
        - 1
        - len(longest_suffix),
        1,
    )
    raise ValueError(
        "Output paths would exceed the Windows MAX_PATH limit before all "
        f"artifacts are written (longest path: {len(str(longest))} characters). "
        f"Use --run-name with at most {max_run_name_length} characters for "
        f"this output directory; current length is {len(run_name)}."
    )


def inverse_softplus(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


def standard_normal_quadrature(point_count: int) -> tuple[np.ndarray, np.ndarray]:
    if point_count < 3:
        raise ValueError("Frailty quadrature requires at least 3 points.")
    nodes, weights = np.polynomial.hermite.hermgauss(point_count)
    nodes = nodes.astype(np.float32) * np.float32(math.sqrt(2.0))
    weights = weights.astype(np.float32) / np.float32(math.sqrt(math.pi))
    return nodes, weights


class LatentRepeatedVisitModel(nn.Module):
    def __init__(
        self,
        availability_feature_count: int,
        detection_feature_count: int,
        species_count: int,
        season_count: int,
        species_season_mode: str,
        detection_frailty_mode: str,
        detection_frailty_init: float,
        frailty_quadrature_points: int,
        initial_availability_logits: np.ndarray,
        initial_detection_logits: np.ndarray,
    ) -> None:
        super().__init__()
        self.species_count = species_count
        self.species_season_mode = species_season_mode
        self.detection_frailty_mode = detection_frailty_mode
        self.availability_weights = nn.Parameter(
            torch.zeros(availability_feature_count, species_count)
        )
        self.availability_bias = nn.Parameter(
            torch.as_tensor(initial_availability_logits.astype(np.float32))
        )
        self.detection_weights = nn.Parameter(
            torch.zeros(detection_feature_count, species_count)
        )
        self.detection_bias = nn.Parameter(
            torch.as_tensor(initial_detection_logits.astype(np.float32))
        )
        if species_season_mode in {"availability", "both"}:
            self.availability_season_bias = nn.Parameter(
                torch.zeros(season_count, species_count)
            )
        else:
            self.availability_season_bias = None
        if species_season_mode in {"detection", "both"}:
            self.detection_season_bias = nn.Parameter(
                torch.zeros(season_count, species_count)
            )
        else:
            self.detection_season_bias = None
        if detection_frailty_mode == "none":
            self.detection_frailty_raw = None
            self.detection_frailty_deviation_raw = None
        elif detection_frailty_mode == "global":
            self.detection_frailty_raw = nn.Parameter(
                torch.full((1,), inverse_softplus(detection_frailty_init))
            )
            self.detection_frailty_deviation_raw = None
        elif detection_frailty_mode == "species":
            self.detection_frailty_raw = nn.Parameter(
                torch.full((species_count,), inverse_softplus(detection_frailty_init))
            )
            self.detection_frailty_deviation_raw = None
        elif detection_frailty_mode == "hierarchical":
            self.detection_frailty_raw = nn.Parameter(
                torch.full((1,), inverse_softplus(detection_frailty_init))
            )
            self.detection_frailty_deviation_raw = nn.Parameter(
                torch.zeros(species_count)
            )
        else:
            raise ValueError(f"Unknown detection frailty mode: {detection_frailty_mode}")
        nodes, weights = standard_normal_quadrature(frailty_quadrature_points)
        self.register_buffer("frailty_nodes", torch.as_tensor(nodes))
        self.register_buffer("frailty_weights", torch.as_tensor(weights))

    def availability_logits(
        self,
        features: torch.Tensor,
        season_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = features @ self.availability_weights + self.availability_bias.unsqueeze(0)
        if self.availability_season_bias is not None:
            if season_index is None:
                raise ValueError("season_index is required for availability season offsets.")
            logits = logits + self.availability_season_bias[season_index]
        return logits

    def detection_logits(
        self,
        features: torch.Tensor,
        season_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = features @ self.detection_weights + self.detection_bias.unsqueeze(0)
        if self.detection_season_bias is not None:
            if season_index is None:
                raise ValueError("season_index is required for detection season offsets.")
            logits = logits + self.detection_season_bias[season_index]
        return logits

    def detection_frailty_scales(self) -> torch.Tensor:
        if self.detection_frailty_raw is None:
            return torch.zeros(
                self.species_count,
                dtype=self.detection_weights.dtype,
                device=self.detection_weights.device,
            )
        raw_scales = self.detection_frailty_raw
        if self.detection_frailty_mode == "hierarchical":
            deviations = self.centered_detection_frailty_deviations()
            raw_scales = raw_scales + deviations
        scales = nn.functional.softplus(raw_scales)
        if self.detection_frailty_mode == "global":
            scales = scales.expand(self.species_count)
        return scales

    def centered_detection_frailty_deviations(self) -> torch.Tensor:
        if self.detection_frailty_deviation_raw is None:
            return torch.zeros(
                self.species_count,
                dtype=self.detection_weights.dtype,
                device=self.detection_weights.device,
            )
        return (
            self.detection_frailty_deviation_raw
            - self.detection_frailty_deviation_raw.mean()
        )

    def mean_detection_probability(self, logits: torch.Tensor) -> torch.Tensor:
        if self.detection_frailty_raw is None:
            return torch.sigmoid(logits)
        scales = self.detection_frailty_scales().unsqueeze(0)
        mean_probability = torch.zeros_like(logits)
        for node, weight in zip(self.frailty_nodes, self.frailty_weights):
            mean_probability = mean_probability + weight * torch.sigmoid(
                logits + node * scales
            )
        return mean_probability

    def species_season_penalty(self) -> torch.Tensor:
        penalty = torch.zeros(
            (), dtype=self.availability_weights.dtype, device=self.availability_weights.device
        )
        terms = 0
        if self.availability_season_bias is not None:
            penalty = penalty + self.availability_season_bias.square().mean()
            terms += 1
        if self.detection_season_bias is not None:
            penalty = penalty + self.detection_season_bias.square().mean()
            terms += 1
        if terms:
            penalty = penalty / terms
        return penalty

    def detection_frailty_penalty(self) -> torch.Tensor:
        if self.detection_frailty_mode == "hierarchical":
            return nn.functional.softplus(self.detection_frailty_raw).square().mean()
        return self.detection_frailty_scales().square().mean()

    def detection_frailty_deviation_penalty(self) -> torch.Tensor:
        return self.centered_detection_frailty_deviations().square().mean()


def sample_groups_for_smoke(
    groups: pd.DataFrame,
    test_season_year: int,
    max_groups_per_split: int | None,
) -> pd.DataFrame:
    if max_groups_per_split is None:
        return groups
    rng = np.random.default_rng(SEED)
    pieces = []
    for mask in [
        groups["season_year"].astype(int) < test_season_year,
        groups["season_year"].astype(int) == test_season_year,
    ]:
        subset = groups.loc[mask]
        if len(subset) > max_groups_per_split:
            subset = subset.sample(
                n=max_groups_per_split,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
        pieces.append(subset)
    selected = pd.concat(pieces, ignore_index=True)
    if selected.empty:
        raise ValueError("Group smoke sample is empty.")
    return selected.sort_values("locality_season_id").reset_index(drop=True)


def build_groups(checklists: pd.DataFrame) -> pd.DataFrame:
    grouped = checklists.groupby("locality_season_id", observed=True, sort=True)
    groups = grouped.agg(
        season_year=("season_year", "first"),
        season_name=("season_name", "first"),
        n_checklists=("sampling_event_identifier", "nunique"),
        n_dates=("n_dates", "first"),
        unique_observers=("unique_observers", "first"),
        duration_bin_count=("duration_bin_count", "first"),
        protocol_count=("protocol_count", "first"),
        canopy_median=("canopy_median", "first"),
        elevation_median=("elevation_median", "first"),
        distance_to_waterbody_m_median=("distance_to_waterbody_m_median", "first"),
        distance_to_coastline_m_median=("distance_to_coastline_m_median", "first"),
    )
    groups["x"] = grouped.geometry.apply(lambda s: float(s.x.median()))
    groups["y"] = grouped.geometry.apply(lambda s: float(s.y.median()))
    return groups.reset_index()


def filter_groups_by_support(
    groups: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    thresholds = {
        "n_checklists": args.min_group_checklists,
        "n_dates": args.min_group_dates,
        "duration_bin_count": args.min_group_duration_bins,
        "protocol_count": args.min_group_protocols,
        "unique_observers": args.min_group_observers,
    }
    mask = np.ones(len(groups), dtype=bool)
    active = {}
    for column, threshold in thresholds.items():
        if threshold is not None:
            if threshold < 1:
                raise ValueError(f"Support threshold for {column} must be at least 1.")
            mask &= groups[column].to_numpy() >= threshold
            active[column] = int(threshold)
    if not active:
        return groups
    filtered = groups.loc[mask].copy()
    if filtered.empty:
        raise ValueError(f"Support filters removed every locality-season: {active}")
    print(
        f"Stricter support filters retained {len(filtered):,} of "
        f"{len(groups):,} locality-seasons: {active}"
    )
    return filtered.reset_index(drop=True)


def make_group_indices(
    checklists: pd.DataFrame,
    groups: pd.DataFrame,
) -> np.ndarray:
    group_index = pd.Series(
        np.arange(len(groups), dtype=np.int64),
        index=groups["locality_season_id"],
    )
    mapped = checklists["locality_season_id"].map(group_index)
    if mapped.isna().any():
        raise ValueError("Some checklists do not map to retained locality-season groups.")
    return mapped.to_numpy(dtype=np.int64)


def make_season_indices(
    groups: pd.DataFrame,
    checklist_group_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    season_names = sorted(groups["season_name"].astype(str).unique())
    season_to_index = {season_name: idx for idx, season_name in enumerate(season_names)}
    group_season_index = (
        groups["season_name"].astype(str).map(season_to_index).to_numpy(dtype=np.int64)
    )
    checklist_season_index = group_season_index[checklist_group_index]
    return group_season_index, checklist_season_index, {"season_names": season_names}


def group_detection_counts(
    labels: np.ndarray,
    checklist_group_index: np.ndarray,
    group_count: int,
) -> np.ndarray:
    counts = np.zeros((group_count, labels.shape[1]), dtype=np.float32)
    np.add.at(counts, checklist_group_index, labels)
    return counts


def initial_availability_logits(
    positive_groups: np.ndarray,
    train_group_mask: np.ndarray,
    smoothing: float = 1.0,
) -> np.ndarray:
    positives = positive_groups[train_group_mask].sum(axis=0)
    trials = train_group_mask.sum()
    rates = (positives + smoothing) / (trials + 2.0 * smoothing)
    rates = np.clip(rates, 1e-5, 1.0 - 1e-5)
    return logit(rates).astype(np.float32)


def remap_train_group_indices(
    checklist_group_index: np.ndarray,
    train_checklist_mask: np.ndarray,
    train_group_indices: np.ndarray,
    group_count: int,
) -> np.ndarray:
    mapping = np.full(group_count, -1, dtype=np.int64)
    mapping[train_group_indices] = np.arange(len(train_group_indices), dtype=np.int64)
    local = mapping[checklist_group_index[train_checklist_mask]]
    if (local < 0).any():
        raise ValueError("Train checklist references a non-train group.")
    return local


def latent_negative_log_likelihood(
    model: LatentRepeatedVisitModel,
    availability_features: torch.Tensor,
    detection_features: torch.Tensor,
    labels: torch.Tensor,
    checklist_group_index: torch.Tensor,
    group_season_index: torch.Tensor,
    checklist_season_index: torch.Tensor,
    positive_groups: torch.Tensor,
) -> torch.Tensor:
    psi_logits = model.availability_logits(availability_features, group_season_index)
    base_detection_logits = model.detection_logits(
        detection_features, checklist_season_index
    )

    log_psi = nn.functional.logsigmoid(psi_logits)
    log_not_psi = nn.functional.logsigmoid(-psi_logits)

    group_count = availability_features.shape[0]
    species_count = labels.shape[1]
    if model.detection_frailty_raw is None:
        log_p = nn.functional.logsigmoid(base_detection_logits)
        log_not_p = nn.functional.logsigmoid(-base_detection_logits)

        sum_log_not_p = torch.zeros(
            group_count, species_count, dtype=labels.dtype, device=labels.device
        )
        sum_log_not_p.index_add_(0, checklist_group_index, log_not_p)

        detection_contrast = labels * (log_p - log_not_p)
        sum_detection_contrast = torch.zeros_like(sum_log_not_p)
        sum_detection_contrast.index_add_(0, checklist_group_index, detection_contrast)
        available_log_likelihood = sum_log_not_p + sum_detection_contrast
        missed_log_likelihood = sum_log_not_p
    else:
        scales = model.detection_frailty_scales().unsqueeze(0)
        available_terms = []
        missed_terms = []
        for node, weight in zip(model.frailty_nodes, model.frailty_weights):
            detection_logits = base_detection_logits + node * scales
            log_p = nn.functional.logsigmoid(detection_logits)
            log_not_p = nn.functional.logsigmoid(-detection_logits)
            sum_log_not_p = torch.zeros(
                group_count, species_count, dtype=labels.dtype, device=labels.device
            )
            sum_log_not_p.index_add_(0, checklist_group_index, log_not_p)
            detection_contrast = labels * (log_p - log_not_p)
            sum_detection_contrast = torch.zeros_like(sum_log_not_p)
            sum_detection_contrast.index_add_(0, checklist_group_index, detection_contrast)
            log_weight = torch.log(weight)
            available_terms.append(sum_log_not_p + sum_detection_contrast + log_weight)
            missed_terms.append(sum_log_not_p + log_weight)
        available_log_likelihood = torch.logsumexp(
            torch.stack(available_terms, dim=0), dim=0
        )
        missed_log_likelihood = torch.logsumexp(torch.stack(missed_terms, dim=0), dim=0)

    positive_log_likelihood = log_psi + available_log_likelihood
    zero_log_likelihood = torch.logaddexp(
        log_not_psi, log_psi + missed_log_likelihood
    )
    log_likelihood = torch.where(
        positive_groups, positive_log_likelihood, zero_log_likelihood
    )
    return -log_likelihood.mean()


def marginal_rate_penalty(
    model: LatentRepeatedVisitModel,
    availability_features: torch.Tensor,
    detection_features: torch.Tensor,
    labels: torch.Tensor,
    checklist_group_index: torch.Tensor,
    group_season_index: torch.Tensor,
    checklist_season_index: torch.Tensor,
    marginal_rate_l2: float,
    species_marginal_rate_l2: float,
) -> torch.Tensor:
    if marginal_rate_l2 <= 0.0 and species_marginal_rate_l2 <= 0.0:
        return torch.zeros((), dtype=labels.dtype, device=labels.device)

    psi = torch.sigmoid(model.availability_logits(availability_features, group_season_index))
    detection_logits = model.detection_logits(detection_features, checklist_season_index)
    conditional_detection = model.mean_detection_probability(detection_logits)
    marginal_detection = conditional_detection * psi[checklist_group_index]

    penalty = torch.zeros((), dtype=labels.dtype, device=labels.device)
    if marginal_rate_l2 > 0.0:
        penalty = penalty + marginal_rate_l2 * (
            marginal_detection.mean() - labels.mean()
        ).square()
    if species_marginal_rate_l2 > 0.0:
        penalty = penalty + species_marginal_rate_l2 * (
            marginal_detection.mean(dim=0) - labels.mean(dim=0)
        ).square().mean()
    return penalty


def fit_latent_model(
    availability_features: np.ndarray,
    detection_features: np.ndarray,
    labels: np.ndarray,
    checklist_group_index: np.ndarray,
    group_season_index: np.ndarray,
    checklist_season_index: np.ndarray,
    positive_groups: np.ndarray,
    train_group_mask: np.ndarray,
    train_checklist_mask: np.ndarray,
    args: argparse.Namespace,
) -> LatentRepeatedVisitModel:
    torch.manual_seed(SEED)
    train_group_indices = np.flatnonzero(train_group_mask)
    train_checklist_group_index = remap_train_group_indices(
        checklist_group_index,
        train_checklist_mask,
        train_group_indices,
        len(train_group_mask),
    )
    model = LatentRepeatedVisitModel(
        availability_features.shape[1],
        detection_features.shape[1],
        labels.shape[1],
        int(group_season_index.max()) + 1,
        args.species_season_mode,
        args.detection_frailty_mode,
        args.detection_frailty_init,
        args.frailty_quadrature_points,
        initial_availability_logits(positive_groups, train_group_mask),
        species_detection_logits(labels, train_checklist_mask),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    availability_tensor = torch.from_numpy(
        availability_features[train_group_mask].astype(np.float32)
    )
    detection_tensor = torch.from_numpy(
        detection_features[train_checklist_mask].astype(np.float32)
    )
    labels_tensor = torch.from_numpy(labels[train_checklist_mask].astype(np.float32))
    group_index_tensor = torch.from_numpy(train_checklist_group_index.astype(np.int64))
    group_season_tensor = torch.from_numpy(
        group_season_index[train_group_mask].astype(np.int64)
    )
    checklist_season_tensor = torch.from_numpy(
        checklist_season_index[train_checklist_mask].astype(np.int64)
    )
    positive_tensor = torch.from_numpy(positive_groups[train_group_mask].astype(bool))

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        nll = latent_negative_log_likelihood(
            model,
            availability_tensor,
            detection_tensor,
            labels_tensor,
            group_index_tensor,
            group_season_tensor,
            checklist_season_tensor,
            positive_tensor,
        )
        objective = nll
        if args.availability_l2 > 0.0:
            objective = objective + args.availability_l2 * model.availability_weights.square().mean()
        if args.detection_l2 > 0.0:
            objective = objective + args.detection_l2 * model.detection_weights.square().mean()
        if args.species_season_l2 > 0.0 and args.species_season_mode != "none":
            objective = objective + args.species_season_l2 * model.species_season_penalty()
        if args.detection_frailty_l2 > 0.0 and args.detection_frailty_mode != "none":
            objective = (
                objective
                + args.detection_frailty_l2 * model.detection_frailty_penalty()
            )
        if (
            args.detection_frailty_deviation_l2 > 0.0
            and args.detection_frailty_mode == "hierarchical"
        ):
            objective = (
                objective
                + args.detection_frailty_deviation_l2
                * model.detection_frailty_deviation_penalty()
            )
        rate_penalty = marginal_rate_penalty(
            model,
            availability_tensor,
            detection_tensor,
            labels_tensor,
            group_index_tensor,
            group_season_tensor,
            checklist_season_tensor,
            args.marginal_rate_l2,
            args.species_marginal_rate_l2,
        )
        objective = objective + rate_penalty
        objective.backward()
        optimizer.step()
        if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
            print(
                f"latent epoch {epoch:>4}: train NLL={float(nll.detach()):.5f}, "
                f"objective={float(objective.detach()):.5f}, "
                f"rate_penalty={float(rate_penalty.detach()):.5f}"
            )
    return model


def predict_latent(
    model: LatentRepeatedVisitModel,
    availability_features: np.ndarray,
    detection_features: np.ndarray,
    checklist_group_index: np.ndarray,
    group_season_index: np.ndarray,
    checklist_season_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        availability_tensor = torch.from_numpy(availability_features.astype(np.float32))
        detection_tensor = torch.from_numpy(detection_features.astype(np.float32))
        group_season_tensor = torch.from_numpy(group_season_index.astype(np.int64))
        checklist_season_tensor = torch.from_numpy(checklist_season_index.astype(np.int64))
        psi = torch.sigmoid(
            model.availability_logits(availability_tensor, group_season_tensor)
        ).numpy()
        detection_logits = model.detection_logits(
            detection_tensor,
            checklist_season_tensor,
        )
        conditional_detection = model.mean_detection_probability(detection_logits).numpy()
    marginal_detection = conditional_detection * psi[checklist_group_index]
    return psi.astype(np.float32), conditional_detection.astype(np.float32), marginal_detection.astype(np.float32)


def predict_group_detection_components(
    model: LatentRepeatedVisitModel,
    detection_features: np.ndarray,
    checklist_group_index: np.ndarray,
    checklist_season_index: np.ndarray,
    group_count: int,
    species_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return model-implied P(any detection | available) and ordered pairs."""
    model.eval()
    group_index = checklist_group_index.astype(np.int64)
    with torch.no_grad():
        detection_tensor = torch.from_numpy(detection_features.astype(np.float32))
        checklist_season_tensor = torch.from_numpy(checklist_season_index.astype(np.int64))
        base_logits = model.detection_logits(detection_tensor, checklist_season_tensor)
        conditional_any = np.zeros((group_count, species_count), dtype=np.float64)
        conditional_pair_counts = np.zeros((group_count, species_count), dtype=np.float64)
        if model.detection_frailty_raw is None:
            nodes = [0.0]
            weights = [1.0]
            scales = torch.zeros((1, species_count), dtype=base_logits.dtype)
        else:
            nodes = [float(value) for value in model.frailty_nodes.cpu().numpy()]
            weights = [float(value) for value in model.frailty_weights.cpu().numpy()]
            scales = model.detection_frailty_scales().unsqueeze(0)

        for node, weight in zip(nodes, weights):
            p = torch.sigmoid(base_logits + node * scales).cpu().numpy().astype(np.float64)
            sum_log_not_p = np.zeros((group_count, species_count), dtype=np.float64)
            np.add.at(sum_log_not_p, group_index, np.log1p(-np.clip(p, EPS, 1.0 - EPS)))
            conditional_any += weight * (-np.expm1(sum_log_not_p))

            sum_p = np.zeros((group_count, species_count), dtype=np.float64)
            sum_p_squared = np.zeros((group_count, species_count), dtype=np.float64)
            np.add.at(sum_p, group_index, p)
            np.add.at(sum_p_squared, group_index, np.square(p))
            conditional_pair_counts += weight * np.maximum(
                np.square(sum_p) - sum_p_squared,
                0.0,
            )
    return conditional_any.astype(np.float32), conditional_pair_counts.astype(np.float32)


def summarize_availability_overall(
    groups: pd.DataFrame,
    group_mask: np.ndarray,
    positive_groups: np.ndarray,
    psi: np.ndarray,
) -> pd.DataFrame:
    y = positive_groups[group_mask].astype(float).ravel()
    p = psi[group_mask].astype(float).ravel()
    ece, max_error = calibration_summary(y, p, 10)
    return pd.DataFrame(
        [
            {
                "model": "latent_availability",
                "locality_seasons": int(group_mask.sum()),
                "pairs": int(y.size),
                "positive_pairs": int(y.sum()),
                "observed_positive_rate": float(y.mean()),
                "mean_predicted_availability": float(p.mean()),
                "calibration_error_vs_observed_positive": abs(float(p.mean()) - float(y.mean())),
                "positive_triplet_auroc": auc_roc(y, p),
                "positive_triplet_auprc": average_precision(y, p),
                "ece_vs_observed_positive": ece,
                "max_bin_error_vs_observed_positive": max_error,
            }
        ]
    )


def summarize_availability_by_species(
    species: pd.DataFrame,
    group_mask: np.ndarray,
    positive_groups: np.ndarray,
    psi: np.ndarray,
) -> pd.DataFrame:
    rows = []
    y_all = positive_groups[group_mask]
    p_all = psi[group_mask]
    for _, row in species.iterrows():
        idx = int(row["species_index"])
        y = y_all[:, idx].astype(float)
        p = p_all[:, idx].astype(float)
        rows.append(
            {
                "species_index": idx,
                "species_key": row["species_key"],
                "common_name": row["common_name"],
                "scientific_name": row["scientific_name"],
                "locality_seasons": int(len(y)),
                "positive_locality_seasons": int(y.sum()),
                "observed_positive_rate": float(y.mean()),
                "mean_predicted_availability": float(p.mean()),
                "calibration_error_vs_observed_positive": abs(float(p.mean()) - float(y.mean())),
                "positive_triplet_auroc": auc_roc(y, p),
                "positive_triplet_auprc": average_precision(y, p),
            }
        )
    return pd.DataFrame(rows)


def posterior_availability(
    psi: np.ndarray,
    conditional_detection: np.ndarray,
    labels: np.ndarray,
    checklist_group_index: np.ndarray,
    positive_groups: np.ndarray,
    missed_if_available: np.ndarray | None = None,
) -> np.ndarray:
    if missed_if_available is None:
        p = np.clip(conditional_detection, EPS, 1.0 - EPS)
        sum_log_not_p = np.zeros_like(psi, dtype=np.float64)
        np.add.at(sum_log_not_p, checklist_group_index, np.log1p(-p))
        missed_if_available = np.exp(sum_log_not_p)
    else:
        missed_if_available = np.clip(missed_if_available.astype(np.float64), EPS, 1.0)
    numerator = psi * missed_if_available
    denominator = (1.0 - psi) + numerator
    posterior = np.divide(
        numerator,
        np.clip(denominator, EPS, None),
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    posterior = posterior.astype(np.float32)
    posterior[positive_groups] = 1.0
    return posterior


def flat_pair_metrics(
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    calibration_bins: int,
) -> dict:
    y = y_true.astype(float).ravel()
    p = scores.astype(float).ravel()
    ece, max_error = calibration_summary(y, p, calibration_bins)
    return {
        "model": model_name,
        "pairs": int(y.size),
        "detections": int(y.sum()),
        "observed_detection_rate": float(y.mean()),
        "mean_predicted_detection_rate": float(p.mean()),
        "calibration_error": abs(float(p.mean()) - float(y.mean())),
        "bce": float(
            -(
                y * np.log(np.clip(p, EPS, 1.0 - EPS))
                + (1.0 - y) * np.log1p(-np.clip(p, EPS, 1.0 - EPS))
            ).mean()
        ),
        "micro_auroc": auc_roc(y, p),
        "micro_auprc": average_precision(y, p),
        "ece": ece,
        "max_bin_error": max_error,
    }


def summarize_latent_detection_diagnostics(
    labels: np.ndarray,
    test_checklist_mask: np.ndarray,
    checklist_group_index: np.ndarray,
    positive_groups: np.ndarray,
    conditional_detection: np.ndarray,
    marginal_detection: np.ndarray,
    posterior: np.ndarray,
    calibration_bins: int,
) -> pd.DataFrame:
    test_labels = labels[test_checklist_mask]
    test_groups = checklist_group_index[test_checklist_mask]
    test_positive_groups = positive_groups[test_groups]
    posterior_marginal = conditional_detection[test_checklist_mask] * posterior[
        test_groups
    ]
    rows = [
        flat_pair_metrics(
            "latent_marginal_all_pairs",
            test_labels,
            marginal_detection[test_checklist_mask],
            calibration_bins,
        ),
        flat_pair_metrics(
            "latent_posterior_marginal_all_pairs_label_informed",
            test_labels,
            posterior_marginal,
            calibration_bins,
        ),
    ]
    if test_positive_groups.any():
        rows.append(
            flat_pair_metrics(
                "latent_conditional_detection_known_available_pairs",
                test_labels[test_positive_groups],
                conditional_detection[test_checklist_mask][test_positive_groups],
                calibration_bins,
            )
        )
    return pd.DataFrame(rows)


def group_detection_probabilities(
    psi: np.ndarray,
    conditional_detection: np.ndarray,
    checklist_group_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return P(at least one detection | available) and its prior marginal."""
    p = np.clip(conditional_detection, EPS, 1.0 - EPS)
    sum_log_not_p = np.zeros_like(psi, dtype=np.float64)
    np.add.at(sum_log_not_p, checklist_group_index, np.log1p(-p))
    conditional_any_detection = -np.expm1(sum_log_not_p)
    prior_any_detection = psi * conditional_any_detection
    return (
        conditional_any_detection.astype(np.float32),
        prior_any_detection.astype(np.float32),
    )


def group_pair_codetection_components(
    groups: pd.DataFrame,
    detection_counts: np.ndarray,
    psi: np.ndarray,
    conditional_pair_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return observed and model-implied ordered pair co-detection quantities."""
    observed_pair_counts = detection_counts.astype(np.float64) * (
        detection_counts.astype(np.float64) - 1.0
    )
    predicted_pair_counts = psi.astype(np.float64) * conditional_pair_counts
    n_checklists = groups["n_checklists"].to_numpy(dtype=np.float64)
    pair_denominator = n_checklists * (n_checklists - 1.0)
    if (pair_denominator <= 0.0).any():
        raise ValueError("Pair co-detection diagnostics require at least two checklists per group.")
    observed_pair_rates = observed_pair_counts / pair_denominator[:, None]
    predicted_pair_rates = predicted_pair_counts / pair_denominator[:, None]
    return (
        observed_pair_counts.astype(np.float32),
        predicted_pair_counts.astype(np.float32),
        observed_pair_rates.astype(np.float32),
        predicted_pair_rates.astype(np.float32),
        pair_denominator.astype(np.float32),
    )


def pair_codetection_metric_row(
    group_mask: np.ndarray,
    observed_pair_counts: np.ndarray,
    predicted_pair_counts: np.ndarray,
    observed_pair_rates: np.ndarray,
    predicted_pair_rates: np.ndarray,
    pair_denominator: np.ndarray,
    species_index: int | None = None,
) -> dict | None:
    group_indices = np.flatnonzero(group_mask)
    if group_indices.size == 0:
        return None
    if species_index is None:
        observed_counts = observed_pair_counts[group_indices].astype(float)
        predicted_counts = predicted_pair_counts[group_indices].astype(float)
        observed_rates = observed_pair_rates[group_indices].astype(float)
        predicted_rates = predicted_pair_rates[group_indices].astype(float)
        possible_pairs = float(pair_denominator[group_indices].sum()) * float(
            observed_pair_counts.shape[1]
        )
    else:
        observed_counts = observed_pair_counts[
            group_indices, species_index
        ].astype(float)
        predicted_counts = predicted_pair_counts[
            group_indices, species_index
        ].astype(float)
        observed_rates = observed_pair_rates[
            group_indices, species_index
        ].astype(float)
        predicted_rates = predicted_pair_rates[
            group_indices, species_index
        ].astype(float)
        possible_pairs = float(pair_denominator[group_indices].sum())

    observed_weighted = float(observed_counts.sum() / possible_pairs)
    predicted_weighted = float(predicted_counts.sum() / possible_pairs)
    observed_unweighted = float(observed_rates.mean())
    predicted_unweighted = float(predicted_rates.mean())
    return {
        "locality_seasons": int(group_indices.size),
        "group_species_pairs": int(observed_rates.size),
        "ordered_checklist_pairs": int(possible_pairs),
        "observed_pair_codetection_rate_weighted": observed_weighted,
        "mean_predicted_pair_codetection_probability_weighted": predicted_weighted,
        "pair_codetection_signed_error_weighted": (
            predicted_weighted - observed_weighted
        ),
        "observed_pair_codetection_rate_group_mean": observed_unweighted,
        "mean_predicted_pair_codetection_probability_group_mean": predicted_unweighted,
        "pair_codetection_signed_error_group_mean": (
            predicted_unweighted - observed_unweighted
        ),
    }


def summarize_pair_codetection_support(
    groups: pd.DataFrame,
    test_group_mask: np.ndarray,
    observed_pair_counts: np.ndarray,
    predicted_pair_counts: np.ndarray,
    observed_pair_rates: np.ndarray,
    predicted_pair_rates: np.ndarray,
    pair_denominator: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for support_type, strata in replication_support_strata(groups).items():
        values = strata.astype("string")
        for stratum in values.dropna().unique():
            group_mask = test_group_mask & values.eq(stratum).fillna(False).to_numpy()
            row = pair_codetection_metric_row(
                group_mask,
                observed_pair_counts,
                predicted_pair_counts,
                observed_pair_rates,
                predicted_pair_rates,
                pair_denominator,
            )
            if row is not None:
                rows.append(
                    {"support_type": support_type, "stratum": str(stratum), **row}
                )
    return pd.DataFrame(rows)


def summarize_pair_codetection_species_season(
    groups: pd.DataFrame,
    species: pd.DataFrame,
    test_group_mask: np.ndarray,
    observed_pair_counts: np.ndarray,
    predicted_pair_counts: np.ndarray,
    observed_pair_rates: np.ndarray,
    predicted_pair_rates: np.ndarray,
    pair_denominator: np.ndarray,
) -> pd.DataFrame:
    rows = []
    season_values = groups["season_name"].astype(str)
    for _, species_row in species.iterrows():
        species_index = int(species_row["species_index"])
        for season_name in sorted(season_values.unique()):
            group_mask = test_group_mask & season_values.eq(season_name).to_numpy()
            row = pair_codetection_metric_row(
                group_mask,
                observed_pair_counts,
                predicted_pair_counts,
                observed_pair_rates,
                predicted_pair_rates,
                pair_denominator,
                species_index,
            )
            if row is not None:
                rows.append(
                    {
                        "species_index": species_index,
                        "species_key": species_row["species_key"],
                        "common_name": species_row["common_name"],
                        "scientific_name": species_row["scientific_name"],
                        "season_name": season_name,
                        **row,
                    }
                )
    return pd.DataFrame(rows)


def component_metric_row(
    group_mask: np.ndarray,
    labels: np.ndarray,
    checklist_group_index: np.ndarray,
    positive_groups: np.ndarray,
    psi: np.ndarray,
    conditional_detection: np.ndarray,
    marginal_detection: np.ndarray,
    conditional_any_detection: np.ndarray,
    prior_any_detection: np.ndarray,
    species_index: int | None = None,
) -> dict | None:
    group_indices = np.flatnonzero(group_mask)
    if group_indices.size == 0:
        return None
    checklist_mask = group_mask[checklist_group_index]
    checklist_groups = checklist_group_index[checklist_mask]
    if checklist_groups.size == 0:
        return None

    if species_index is None:
        group_observed = positive_groups[group_indices].astype(float)
        group_psi = psi[group_indices].astype(float)
        group_conditional_any = conditional_any_detection[group_indices].astype(float)
        group_prior_any = prior_any_detection[group_indices].astype(float)
        checklist_observed = labels[checklist_mask].astype(float)
        checklist_marginal = marginal_detection[checklist_mask].astype(float)
        checklist_conditional = conditional_detection[checklist_mask].astype(float)
        known_positive = positive_groups[checklist_groups]
    else:
        group_observed = positive_groups[group_indices, species_index].astype(float)
        group_psi = psi[group_indices, species_index].astype(float)
        group_conditional_any = conditional_any_detection[
            group_indices, species_index
        ].astype(float)
        group_prior_any = prior_any_detection[group_indices, species_index].astype(float)
        checklist_observed = labels[checklist_mask, species_index].astype(float)
        checklist_marginal = marginal_detection[
            checklist_mask, species_index
        ].astype(float)
        checklist_conditional = conditional_detection[
            checklist_mask, species_index
        ].astype(float)
        known_positive = positive_groups[checklist_groups, species_index]

    observed_any_rate = float(group_observed.mean())
    predicted_any_rate = float(group_prior_any.mean())
    observed_checklist_rate = float(checklist_observed.mean())
    predicted_checklist_rate = float(checklist_marginal.mean())
    known_pairs = int(known_positive.sum())
    if known_pairs:
        known_observed_rate = float(checklist_observed[known_positive].mean())
        known_predicted_rate = float(checklist_conditional[known_positive].mean())
    else:
        known_observed_rate = float("nan")
        known_predicted_rate = float("nan")

    return {
        "locality_seasons": int(group_indices.size),
        "group_species_pairs": int(group_observed.size),
        "observed_any_detection_rate": observed_any_rate,
        "mean_predicted_any_detection_probability": predicted_any_rate,
        "any_detection_signed_error": predicted_any_rate - observed_any_rate,
        "any_detection_absolute_error": abs(predicted_any_rate - observed_any_rate),
        "mean_predicted_availability": float(group_psi.mean()),
        "availability_minus_observed_any_detection": (
            float(group_psi.mean()) - observed_any_rate
        ),
        "mean_conditional_any_detection_if_available": float(
            group_conditional_any.mean()
        ),
        "checklists": int(checklist_mask.sum()),
        "checklist_species_pairs": int(checklist_observed.size),
        "observed_checklist_detection_rate": observed_checklist_rate,
        "mean_prior_marginal_detection_probability": predicted_checklist_rate,
        "prior_marginal_signed_error": (
            predicted_checklist_rate - observed_checklist_rate
        ),
        "prior_marginal_absolute_error": abs(
            predicted_checklist_rate - observed_checklist_rate
        ),
        "known_positive_group_checklist_species_pairs": known_pairs,
        "known_positive_group_observed_detection_rate": known_observed_rate,
        "mean_conditional_detection_known_positive_groups": known_predicted_rate,
        "conditional_detection_signed_error_known_positive_groups": (
            known_predicted_rate - known_observed_rate
        ),
    }


def replication_support_strata(groups: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "checklists": pd.cut(
            groups["n_checklists"],
            bins=[0, 3, 5, 10, np.inf],
            labels=["1-3", "4-5", "6-10", "11+"],
            include_lowest=True,
        ),
        "dates": pd.cut(
            groups["n_dates"],
            bins=[0, 2, 4, 9, np.inf],
            labels=["1-2", "3-4", "5-9", "10+"],
            include_lowest=True,
        ),
        "duration_bins": pd.cut(
            groups["duration_bin_count"],
            bins=[0, 1, 2, np.inf],
            labels=["1", "2", "3+"],
            include_lowest=True,
        ),
        "protocols": pd.cut(
            groups["protocol_count"],
            bins=[0, 1, np.inf],
            labels=["1", "2+"],
            include_lowest=True,
        ),
        "observers": pd.cut(
            groups["unique_observers"],
            bins=[0, 1, 2, 5, np.inf],
            labels=["1", "2", "3-5", "6+"],
            include_lowest=True,
        ),
    }


def summarize_component_support(
    groups: pd.DataFrame,
    test_group_mask: np.ndarray,
    labels: np.ndarray,
    checklist_group_index: np.ndarray,
    positive_groups: np.ndarray,
    psi: np.ndarray,
    conditional_detection: np.ndarray,
    marginal_detection: np.ndarray,
    conditional_any_detection: np.ndarray,
    prior_any_detection: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for support_type, strata in replication_support_strata(groups).items():
        values = strata.astype("string")
        for stratum in values.dropna().unique():
            group_mask = test_group_mask & values.eq(stratum).fillna(False).to_numpy()
            row = component_metric_row(
                group_mask,
                labels,
                checklist_group_index,
                positive_groups,
                psi,
                conditional_detection,
                marginal_detection,
                conditional_any_detection,
                prior_any_detection,
            )
            if row is not None:
                rows.append(
                    {"support_type": support_type, "stratum": str(stratum), **row}
                )
    return pd.DataFrame(rows)


def summarize_component_species_season(
    groups: pd.DataFrame,
    species: pd.DataFrame,
    test_group_mask: np.ndarray,
    labels: np.ndarray,
    checklist_group_index: np.ndarray,
    positive_groups: np.ndarray,
    psi: np.ndarray,
    conditional_detection: np.ndarray,
    marginal_detection: np.ndarray,
    conditional_any_detection: np.ndarray,
    prior_any_detection: np.ndarray,
) -> pd.DataFrame:
    rows = []
    season_values = groups["season_name"].astype(str)
    for _, species_row in species.iterrows():
        species_index = int(species_row["species_index"])
        for season_name in sorted(season_values.unique()):
            group_mask = test_group_mask & season_values.eq(season_name).to_numpy()
            row = component_metric_row(
                group_mask,
                labels,
                checklist_group_index,
                positive_groups,
                psi,
                conditional_detection,
                marginal_detection,
                conditional_any_detection,
                prior_any_detection,
                species_index,
            )
            if row is not None:
                rows.append(
                    {
                        "species_index": species_index,
                        "species_key": species_row["species_key"],
                        "common_name": species_row["common_name"],
                        "scientific_name": species_row["scientific_name"],
                        "season_name": season_name,
                        **row,
                    }
                )
    return pd.DataFrame(rows)


def summarize_component_species_support(
    groups: pd.DataFrame,
    species: pd.DataFrame,
    test_group_mask: np.ndarray,
    labels: np.ndarray,
    checklist_group_index: np.ndarray,
    positive_groups: np.ndarray,
    psi: np.ndarray,
    conditional_detection: np.ndarray,
    marginal_detection: np.ndarray,
    conditional_any_detection: np.ndarray,
    prior_any_detection: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for support_type, strata in replication_support_strata(groups).items():
        values = strata.astype("string")
        for stratum in values.dropna().unique():
            group_mask = test_group_mask & values.eq(stratum).fillna(False).to_numpy()
            for _, species_row in species.iterrows():
                species_index = int(species_row["species_index"])
                row = component_metric_row(
                    group_mask,
                    labels,
                    checklist_group_index,
                    positive_groups,
                    psi,
                    conditional_detection,
                    marginal_detection,
                    conditional_any_detection,
                    prior_any_detection,
                    species_index,
                )
                if row is not None:
                    rows.append(
                        {
                            "species_index": species_index,
                            "species_key": species_row["species_key"],
                            "common_name": species_row["common_name"],
                            "scientific_name": species_row["scientific_name"],
                            "support_type": support_type,
                            "stratum": str(stratum),
                            **row,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_focus_species_availability_season(
    groups: pd.DataFrame,
    species: pd.DataFrame,
    group_mask: np.ndarray,
    positive_groups: np.ndarray,
    psi: np.ndarray,
    focus_species: list[str],
) -> pd.DataFrame:
    focus = species.loc[
        species["common_name"].isin(focus_species),
        ["species_index", "common_name"],
    ]
    if focus.empty:
        return pd.DataFrame()
    rows = []
    test_groups = groups.loc[group_mask].reset_index(drop=True)
    y_all = positive_groups[group_mask]
    p_all = psi[group_mask]
    for _, species_row in focus.iterrows():
        idx = int(species_row["species_index"])
        work = test_groups[["season_name"]].copy()
        work["observed_positive"] = y_all[:, idx].astype(float)
        work["predicted_availability"] = p_all[:, idx].astype(float)
        for season_name, group in work.groupby("season_name", observed=True):
            rows.append(
                {
                    "common_name": species_row["common_name"],
                    "season_name": season_name,
                    "locality_seasons": int(len(group)),
                    "positive_locality_seasons": int(group["observed_positive"].sum()),
                    "observed_positive_rate": float(group["observed_positive"].mean()),
                    "mean_predicted_availability": float(group["predicted_availability"].mean()),
                    "calibration_error_vs_observed_positive": abs(
                        float(group["predicted_availability"].mean())
                        - float(group["observed_positive"].mean())
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_focus_species_group_predictions(
    groups: pd.DataFrame,
    species: pd.DataFrame,
    group_mask: np.ndarray,
    positive_groups: np.ndarray,
    psi: np.ndarray,
    conditional_any_detection: np.ndarray,
    prior_any_detection: np.ndarray,
    focus_species: list[str],
) -> pd.DataFrame:
    focus = species.loc[
        species["common_name"].isin(focus_species),
        ["species_index", "species_key", "common_name", "scientific_name"],
    ]
    if focus.empty:
        return pd.DataFrame()

    group_indices = np.flatnonzero(group_mask)
    test_groups = groups.iloc[group_indices].reset_index(drop=True)
    frames = []
    for _, species_row in focus.iterrows():
        species_index = int(species_row["species_index"])
        frame = test_groups.copy()
        frame.insert(0, "scientific_name", species_row["scientific_name"])
        frame.insert(0, "common_name", species_row["common_name"])
        frame.insert(0, "species_key", species_row["species_key"])
        frame.insert(0, "species_index", species_index)
        frame["observed_any_detection"] = positive_groups[
            group_indices, species_index
        ].astype(np.int8)
        frame["predicted_availability"] = psi[
            group_indices, species_index
        ].astype(float)
        frame["conditional_any_detection_probability"] = conditional_any_detection[
            group_indices, species_index
        ].astype(float)
        frame["prior_any_detection_probability"] = prior_any_detection[
            group_indices, species_index
        ].astype(float)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_outputs(
    output_dir: Path,
    run_name: str,
    metrics: pd.DataFrame,
    species_metrics: pd.DataFrame,
    availability_metrics: pd.DataFrame,
    availability_species_metrics: pd.DataFrame,
    latent_detection_diagnostics: pd.DataFrame,
    focus_detection: pd.DataFrame,
    focus_availability: pd.DataFrame,
    focus_group_predictions: pd.DataFrame,
    component_support: pd.DataFrame,
    component_species_season: pd.DataFrame,
    component_species_support: pd.DataFrame,
    pair_codetection_support: pd.DataFrame,
    pair_codetection_species_season: pd.DataFrame,
    frailty_species: pd.DataFrame,
    summary: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / f"{run_name}_metrics.csv", index=False)
    species_metrics.to_csv(output_dir / f"{run_name}_species_metrics.csv", index=False)
    availability_metrics.to_csv(
        output_dir / f"{run_name}_availability_metrics.csv", index=False
    )
    availability_species_metrics.to_csv(
        output_dir / f"{run_name}_availability_species_metrics.csv", index=False
    )
    latent_detection_diagnostics.to_csv(
        output_dir / f"{run_name}_latent_detection_diagnostics.csv", index=False
    )
    focus_detection.to_csv(
        output_dir / f"{run_name}_focus_species_season.csv", index=False
    )
    focus_availability.to_csv(
        output_dir / f"{run_name}_focus_species_availability_season.csv",
        index=False,
    )
    focus_group_predictions.to_csv(
        output_dir / f"{run_name}_focus_species_group_predictions.csv",
        index=False,
    )
    component_support.to_csv(
        output_dir / f"{run_name}_component_support_metrics.csv", index=False
    )
    component_species_season.to_csv(
        output_dir / f"{run_name}_component_species_season_metrics.csv", index=False
    )
    component_species_support.to_csv(
        output_dir / f"{run_name}_component_species_support_metrics.csv", index=False
    )
    pair_codetection_support.to_csv(
        output_dir / f"{run_name}_pair_codetection_support_metrics.csv", index=False
    )
    pair_codetection_species_season.to_csv(
        output_dir / f"{run_name}_pair_codetection_species_season_metrics.csv",
        index=False,
    )
    frailty_species.to_csv(
        output_dir / f"{run_name}_frailty_species.csv", index=False
    )
    (output_dir / f"{run_name}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(torch.get_num_threads(), 1))
    dataset_dir = Path(args.dataset_dir)
    processed_dir = Path(args.processed_dir)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else dataset_dir / DEFAULT_OUTPUT_DIR_NAME
    )
    validate_output_paths(output_dir, args.run_name)

    metadata = load_metadata(dataset_dir)
    species = load_species(dataset_dir)
    checklists = load_and_assign_checklists(
        processed_dir,
        dataset_dir,
        metadata,
        args.include_inadequate,
        max_checklists=None,
    )
    groups = build_groups(checklists)
    groups = filter_groups_by_support(groups, args)
    groups = sample_groups_for_smoke(
        groups, args.test_season_year, args.max_groups_per_split
    )
    checklists = checklists.loc[
        checklists["locality_season_id"].isin(groups["locality_season_id"])
    ].reset_index(drop=True)
    groups = build_groups(checklists)
    checklist_group_index = make_group_indices(checklists, groups)
    group_season_index, checklist_season_index, season_metadata = make_season_indices(
        groups, checklist_group_index
    )

    group_years = groups["season_year"].astype(int).to_numpy()
    train_group_mask = group_years < args.test_season_year
    test_group_mask = group_years == args.test_season_year
    if not train_group_mask.any() or not test_group_mask.any():
        raise ValueError(
            "Latent split failed: need at least one train and one test locality-season."
        )
    train_checklist_mask = train_group_mask[checklist_group_index]
    test_checklist_mask = test_group_mask[checklist_group_index]

    labels = load_labels(processed_dir, checklists, species)
    detection_counts = group_detection_counts(
        labels, checklist_group_index, len(groups)
    )
    positive_groups = detection_counts > 0.0

    availability_frame = build_feature_frame(
        groups, "availability", args.include_coordinates
    )
    availability_features, availability_metadata = standardize(
        availability_frame, train_group_mask
    )
    effort_frame = build_effort_features(checklists)
    detection_features, detection_metadata = standardize_checklist_features(
        effort_frame, train_checklist_mask
    )

    print(
        "Latent data: "
        f"groups train={int(train_group_mask.sum()):,}, "
        f"test={int(test_group_mask.sum()):,}; "
        f"checklists train={int(train_checklist_mask.sum()):,}, "
        f"test={int(test_checklist_mask.sum()):,}; "
        f"species={len(species)}"
    )

    model = fit_latent_model(
        availability_features,
        detection_features,
        labels,
        checklist_group_index,
        group_season_index,
        checklist_season_index,
        positive_groups,
        train_group_mask,
        train_checklist_mask,
        args,
    )
    psi, conditional_detection, marginal_detection = predict_latent(
        model,
        availability_features,
        detection_features,
        checklist_group_index,
        group_season_index,
        checklist_season_index,
    )
    conditional_any_detection, conditional_pair_counts = (
        predict_group_detection_components(
            model,
            detection_features,
            checklist_group_index,
            checklist_season_index,
            len(groups),
            len(species),
        )
    )
    prior_any_detection = psi * conditional_any_detection
    posterior = posterior_availability(
        psi,
        conditional_detection,
        labels,
        checklist_group_index,
        positive_groups,
        missed_if_available=1.0 - conditional_any_detection,
    )
    (
        observed_pair_counts,
        predicted_pair_counts,
        observed_pair_rates,
        predicted_pair_rates,
        pair_denominator,
    ) = group_pair_codetection_components(
        groups,
        detection_counts,
        psi,
        conditional_pair_counts,
    )

    metrics = pd.DataFrame(
        [
            summarize_overall(
                labels,
                test_checklist_mask,
                marginal_detection,
                "latent_marginal",
                args.calibration_bins,
            )
        ]
    )
    species_metrics = summarize_by_species(
        labels,
        test_checklist_mask,
        marginal_detection,
        species,
        "latent_marginal",
    )
    availability_metrics = summarize_availability_overall(
        groups, test_group_mask, positive_groups, psi
    )
    availability_species_metrics = summarize_availability_by_species(
        species, test_group_mask, positive_groups, psi
    )
    latent_detection_diagnostics = summarize_latent_detection_diagnostics(
        labels,
        test_checklist_mask,
        checklist_group_index,
        positive_groups,
        conditional_detection,
        marginal_detection,
        posterior,
        args.calibration_bins,
    )
    focus_detection = summarize_focus_species_season(
        checklists,
        labels,
        test_checklist_mask,
        {"latent_marginal": marginal_detection},
        species,
        args.focus_species,
    )
    focus_availability = summarize_focus_species_availability_season(
        groups,
        species,
        test_group_mask,
        positive_groups,
        psi,
        args.focus_species,
    )
    focus_group_predictions = build_focus_species_group_predictions(
        groups,
        species,
        test_group_mask,
        positive_groups,
        psi,
        conditional_any_detection,
        prior_any_detection,
        args.focus_species,
    )
    component_support = summarize_component_support(
        groups,
        test_group_mask,
        labels,
        checklist_group_index,
        positive_groups,
        psi,
        conditional_detection,
        marginal_detection,
        conditional_any_detection,
        prior_any_detection,
    )
    component_species_season = summarize_component_species_season(
        groups,
        species,
        test_group_mask,
        labels,
        checklist_group_index,
        positive_groups,
        psi,
        conditional_detection,
        marginal_detection,
        conditional_any_detection,
        prior_any_detection,
    )
    component_species_support = summarize_component_species_support(
        groups,
        species,
        test_group_mask,
        labels,
        checklist_group_index,
        positive_groups,
        psi,
        conditional_detection,
        marginal_detection,
        conditional_any_detection,
        prior_any_detection,
    )
    pair_codetection_support = summarize_pair_codetection_support(
        groups,
        test_group_mask,
        observed_pair_counts,
        predicted_pair_counts,
        observed_pair_rates,
        predicted_pair_rates,
        pair_denominator,
    )
    pair_codetection_species_season = summarize_pair_codetection_species_season(
        groups,
        species,
        test_group_mask,
        observed_pair_counts,
        predicted_pair_counts,
        observed_pair_rates,
        predicted_pair_rates,
        pair_denominator,
    )

    frailty_scales = (
        model.detection_frailty_scales().detach().cpu().numpy().astype(float)
    )
    frailty_species = species[
        ["species_index", "species_key", "common_name", "scientific_name"]
    ].copy()
    frailty_species["detection_frailty_scale"] = frailty_scales

    parameter_summary = {
        "availability_weight_rms": float(
            torch.sqrt(model.availability_weights.detach().square().mean())
        ),
        "availability_bias_rms": float(
            torch.sqrt(model.availability_bias.detach().square().mean())
        ),
        "detection_weight_rms": float(
            torch.sqrt(model.detection_weights.detach().square().mean())
        ),
        "detection_bias_rms": float(
            torch.sqrt(model.detection_bias.detach().square().mean())
        ),
        "availability_season_bias_rms": (
            float(torch.sqrt(model.availability_season_bias.detach().square().mean()))
            if model.availability_season_bias is not None
            else 0.0
        ),
        "detection_season_bias_rms": (
            float(torch.sqrt(model.detection_season_bias.detach().square().mean()))
            if model.detection_season_bias is not None
            else 0.0
        ),
        "detection_frailty_scale_mean": float(np.mean(frailty_scales)),
        "detection_frailty_scale_std": float(np.std(frailty_scales)),
        "detection_frailty_scale_rms": float(
            np.sqrt(np.mean(np.square(frailty_scales)))
        ),
        "detection_frailty_scale_min": float(np.min(frailty_scales)),
        "detection_frailty_scale_q05": float(np.quantile(frailty_scales, 0.05)),
        "detection_frailty_scale_median": float(np.median(frailty_scales)),
        "detection_frailty_scale_q95": float(np.quantile(frailty_scales, 0.95)),
        "detection_frailty_scale_max": float(np.max(frailty_scales)),
        "detection_frailty_deviation_rms": float(
            torch.sqrt(
                model.centered_detection_frailty_deviations().detach().square().mean()
            )
        ),
    }
    summary = {
        "run_name": args.run_name,
        "dataset_dir": str(dataset_dir),
        "processed_dir": str(processed_dir),
        "test_season_year": int(args.test_season_year),
        "epochs": int(args.epochs),
        "groups": {
            "total": int(len(groups)),
            "train": int(train_group_mask.sum()),
            "test": int(test_group_mask.sum()),
        },
        "checklists": {
            "total": int(len(checklists)),
            "train": int(train_checklist_mask.sum()),
            "test": int(test_checklist_mask.sum()),
        },
        "species": int(len(species)),
        "regularization": {
            "availability_l2": float(args.availability_l2),
            "detection_l2": float(args.detection_l2),
            "marginal_rate_l2": float(args.marginal_rate_l2),
            "species_marginal_rate_l2": float(args.species_marginal_rate_l2),
            "species_season_mode": args.species_season_mode,
            "species_season_l2": float(args.species_season_l2),
            "detection_frailty_mode": args.detection_frailty_mode,
            "detection_frailty_init": float(args.detection_frailty_init),
            "detection_frailty_l2": float(args.detection_frailty_l2),
            "detection_frailty_deviation_l2": float(
                args.detection_frailty_deviation_l2
            ),
            "frailty_quadrature_points": int(args.frailty_quadrature_points),
            "weight_decay": float(args.weight_decay),
        },
        "support_filters": {
            "min_group_checklists": args.min_group_checklists,
            "min_group_dates": args.min_group_dates,
            "min_group_duration_bins": args.min_group_duration_bins,
            "min_group_protocols": args.min_group_protocols,
            "min_group_observers": args.min_group_observers,
        },
        "feature_metadata": {
            "availability": availability_metadata,
            "detection": detection_metadata,
            "species_season": season_metadata,
        },
        "parameter_summary": parameter_summary,
        "checklist_metrics": metrics.to_dict(orient="records"),
        "availability_metrics": availability_metrics.to_dict(orient="records"),
        "latent_detection_diagnostics": latent_detection_diagnostics.to_dict(
            orient="records"
        ),
        "component_diagnostic_rows": {
            "support": int(len(component_support)),
            "species_season": int(len(component_species_season)),
            "species_support": int(len(component_species_support)),
            "pair_codetection_support": int(len(pair_codetection_support)),
            "pair_codetection_species_season": int(
                len(pair_codetection_species_season)
            ),
            "focus_species_group_predictions": int(len(focus_group_predictions)),
        },
    }
    write_outputs(
        output_dir,
        args.run_name,
        metrics,
        species_metrics,
        availability_metrics,
        availability_species_metrics,
        latent_detection_diagnostics,
        focus_detection,
        focus_availability,
        focus_group_predictions,
        component_support,
        component_species_season,
        component_species_support,
        pair_codetection_support,
        pair_codetection_species_season,
        frailty_species,
        summary,
    )

    print("\nChecklist-level latent detection metrics:")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.5f}"))
    print("\nGroup-level latent availability diagnostics:")
    print(
        availability_metrics.to_string(
            index=False, float_format=lambda value: f"{value:.5f}"
        )
    )
    print("\nLatent detection diagnostics:")
    print(
        latent_detection_diagnostics.to_string(
            index=False, float_format=lambda value: f"{value:.5f}"
        )
    )
    print(
        "\nComponent diagnostic rows: "
        f"support={len(component_support):,}, "
        f"species-season={len(component_species_season):,}, "
        f"species-support={len(component_species_support):,}"
    )
    print(
        "Detection frailty scales: "
        f"mode={args.detection_frailty_mode}, "
        f"mean={parameter_summary['detection_frailty_scale_mean']:.5f}, "
        f"std={parameter_summary['detection_frailty_scale_std']:.5f}, "
        f"min={parameter_summary['detection_frailty_scale_min']:.5f}, "
        f"median={parameter_summary['detection_frailty_scale_median']:.5f}, "
        f"max={parameter_summary['detection_frailty_scale_max']:.5f}"
    )
    support_columns = [
        "support_type",
        "stratum",
        "locality_seasons",
        "observed_any_detection_rate",
        "mean_predicted_any_detection_probability",
        "any_detection_signed_error",
        "observed_checklist_detection_rate",
        "mean_prior_marginal_detection_probability",
        "prior_marginal_signed_error",
    ]
    print("\nComponent diagnostics by replication support:")
    print(
        component_support[support_columns].to_string(
            index=False, float_format=lambda value: f"{value:.5f}"
        )
    )
    season_error_columns = [
        "common_name",
        "season_name",
        "locality_seasons",
        "observed_any_detection_rate",
        "mean_predicted_any_detection_probability",
        "any_detection_signed_error",
        "observed_checklist_detection_rate",
        "mean_prior_marginal_detection_probability",
        "prior_marginal_signed_error",
    ]
    largest_season_errors = component_species_season.assign(
        _abs_error=component_species_season["prior_marginal_signed_error"].abs()
    ).nlargest(15, "_abs_error")
    print("\nLargest species-season fair prior-marginal errors:")
    print(
        largest_season_errors[season_error_columns].to_string(
            index=False, float_format=lambda value: f"{value:.5f}"
        )
    )
    pair_columns = [
        "support_type",
        "stratum",
        "locality_seasons",
        "observed_pair_codetection_rate_weighted",
        "mean_predicted_pair_codetection_probability_weighted",
        "pair_codetection_signed_error_weighted",
        "observed_pair_codetection_rate_group_mean",
        "mean_predicted_pair_codetection_probability_group_mean",
        "pair_codetection_signed_error_group_mean",
    ]
    print("\nPairwise co-detection diagnostics by replication support:")
    print(
        pair_codetection_support[pair_columns].to_string(
            index=False, float_format=lambda value: f"{value:.5f}"
        )
    )
    print(f"\nWrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
