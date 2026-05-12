"""
Spatial IPPP experiment for Wood Thrush eBird observations.

This mirrors the spatial portion of exp/nippp.py:

- HPPP closed-form baseline
- Constant neural IPPP sanity check
- Linear IPPP with x/y trend
- likelihood-ratio test
- Berman-Turner residual diagnostics
- grid-resolution sensitivity

This version intentionally does not use raster covariates or temporal terms yet.
The observation window can be an irregular boundary dataset, a WGS84 bbox, or
the rectangular extent of the observations.

Run from the project root:

    python exp/wood_thrush_nippp.py --input data/wood_thrush_nc_2020_2023.geojson
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import chi2
from shapely.geometry import Point, box


SEED = 19
DEFAULT_INPUT = "data/wood_thrush_nc_2020_2023.geojson"
DEFAULT_IMAGE_DIR = "images/wood_thrush_nippp"
MIN_LAMBDA = 1e-30
MIN_EXPECTED_COUNT = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the baseline spatial IPPP experiment to Wood Thrush observations."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input Wood Thrush point GeoJSON. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--image-dir",
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory for output figures. Defaults to {DEFAULT_IMAGE_DIR}.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="Berman-Turner grid cells per dimension for the main fit. Defaults to 100.",
    )
    parser.add_argument(
        "--plot-grid-size",
        type=int,
        default=200,
        help="Grid cells per dimension for intensity surface plotting. Defaults to 200.",
    )
    parser.add_argument(
        "--epochs-constant",
        type=int,
        default=3000,
        help="Training epochs for the constant neural IPPP. Defaults to 3000.",
    )
    parser.add_argument(
        "--epochs-linear",
        type=int,
        default=10000,
        help="Training epochs for the linear IPPP. Defaults to 10000.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate. Defaults to 1e-3.",
    )
    parser.add_argument(
        "--grid-sensitivity",
        nargs="+",
        type=int,
        default=[50, 75, 100, 150, 200],
        help="Grid sizes for sensitivity analysis. Defaults to 50 75 100 150 200.",
    )
    parser.add_argument("--south", type=float, help="Optional WGS84 window south coordinate.")
    parser.add_argument("--north", type=float, help="Optional WGS84 window north coordinate.")
    parser.add_argument("--west", type=float, help="Optional WGS84 window west coordinate.")
    parser.add_argument("--east", type=float, help="Optional WGS84 window east coordinate.")
    parser.add_argument(
        "--boundary",
        help=(
            "Optional study-window boundary readable by GeoPandas, such as a "
            "shapefile, GeoPackage, or GeoJSON. Points outside it are dropped "
            "and quadrature cell weights use intersection area."
        ),
    )
    parser.add_argument(
        "--analysis-crs",
        help=(
            "Optional CRS for fitting after reading inputs, e.g. EPSG:5070. "
            "An equal-area CRS is preferred because the IPPP likelihood uses area."
        ),
    )
    parser.add_argument(
        "--plot-crs",
        default="EPSG:4326",
        help="CRS used only for map figures. Defaults to EPSG:4326 for longitude/latitude axes.",
    )
    parser.add_argument(
        "--cv-blocks-per-dim",
        type=int,
        default=5,
        help="Number of spatial blocks per x/y dimension for block cross-validation. Defaults to 5.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of spatial block CV folds. Defaults to 5.",
    )
    parser.add_argument(
        "--simulation-count",
        type=int,
        default=500,
        help="Number of fitted-IPPP simulations for simulation diagnostics. Defaults to 500.",
    )
    parser.add_argument(
        "--k-radii",
        type=int,
        default=50,
        help="Number of radii for the inhomogeneous K-function diagnostic. Defaults to 50.",
    )
    return parser.parse_args()


def validate_bbox(args: argparse.Namespace) -> tuple[float, float, float, float] | None:
    values = (args.south, args.north, args.west, args.east)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("--south, --north, --west, and --east must be provided together.")
    if args.south >= args.north:
        raise ValueError("--south must be less than --north.")
    if args.west >= args.east:
        raise ValueError("--west must be less than --east.")
    return args.south, args.north, args.west, args.east


def load_points(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Input point dataset has no rows: {path}")
    if gdf.crs is None:
        raise ValueError(f"Input point dataset has no CRS: {path}")

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.geom_type == "Point"].copy()
    if gdf.empty:
        raise ValueError("Input dataset has no valid point geometries.")
    return gdf


def dissolve_geometry(gdf: gpd.GeoDataFrame):
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def load_boundary(path: Path, target_crs) -> object:
    if not path.exists():
        raise FileNotFoundError(f"Boundary file does not exist: {path}")

    boundary = gpd.read_file(path)
    if boundary.empty:
        raise ValueError(f"Boundary file has no features: {path}")
    if boundary.crs is None:
        raise ValueError(f"Boundary file has no CRS: {path}")

    boundary = boundary[boundary.geometry.notna()].copy()
    boundary = boundary[~boundary.geometry.is_empty].copy()
    if boundary.empty:
        raise ValueError(f"Boundary file has no valid geometries: {path}")

    boundary = boundary.to_crs(target_crs)
    window_geom = dissolve_geometry(boundary)
    if window_geom.is_empty:
        raise ValueError(f"Boundary dissolved to an empty geometry: {path}")
    return window_geom


def bbox_geometry(
    bbox: tuple[float, float, float, float],
    target_crs,
) -> object:
    south, north, west, east = bbox
    bbox_gdf = gpd.GeoDataFrame(
        geometry=[box(west, south, east, north)],
        crs="EPSG:4326",
    ).to_crs(target_crs)
    return bbox_gdf.geometry.iloc[0]


def build_window_geometry(
    points: gpd.GeoDataFrame,
    boundary_path: str | None,
    bbox: tuple[float, float, float, float] | None,
) -> object:
    window_geom = None
    if boundary_path is not None:
        window_geom = load_boundary(Path(boundary_path), points.crs)

    if bbox is not None:
        bbox_geom = bbox_geometry(bbox, points.crs)
        window_geom = bbox_geom if window_geom is None else window_geom.intersection(bbox_geom)

    if window_geom is None:
        west, south, east, north = points.total_bounds
        window_geom = box(west, south, east, north)

    if window_geom.is_empty:
        raise ValueError("Study window geometry is empty.")
    if window_geom.area <= 0:
        raise ValueError("Study window geometry has zero area.")
    return window_geom


def filter_points_to_window(
    points: gpd.GeoDataFrame,
    window_geom,
    should_filter: bool,
) -> gpd.GeoDataFrame:
    if not should_filter:
        return points

    filtered = points.loc[points.geometry.intersects(window_geom)].copy()
    if filtered.empty:
        raise ValueError("No observations fall inside the requested study window.")
    print(f"Study window kept {len(filtered):,} of {len(points):,} observations")
    return filtered


class ConstantIPPP(nn.Module):
    """Constant-intensity IPPP: lambda(s) = lambda0."""

    def __init__(self, lambda_init: float):
        super().__init__()
        self.log_lambda = nn.Parameter(
            torch.tensor(np.log(lambda_init), dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lambda0 = torch.exp(self.log_lambda)
        return lambda0.expand(x.shape[0], 1)


class LinearIPPP(nn.Module):
    """Log-linear spatial IPPP: lambda(s) = exp(beta0 + beta1*x + beta2*y)."""

    def __init__(self, input_dim: int, lambda_init: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        with torch.no_grad():
            self.linear.weight.zero_()
            self.linear.bias.fill_(np.log(lambda_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eta = self.linear(x)
        eta = torch.clamp(eta, min=-20, max=20)
        return torch.exp(eta)


def aic(ll: float, k: int) -> float:
    return 2 * k - 2 * ll


def bic(ll: float, k: int, n: int) -> float:
    return k * np.log(n) - 2 * ll


def lr_stat(ll_full: float, ll_reduced: float) -> float:
    return 2 * (ll_full - ll_reduced)


class WoodThrushExperiment:
    def __init__(
        self,
        coords_np: np.ndarray,
        window_geom,
        grid_size: int,
    ):
        self.coords_np = coords_np.astype(np.float64)
        self.window_geom = window_geom
        self.x_min, self.y_min, self.x_max, self.y_max = window_geom.bounds
        self.grid_size = grid_size
        self.n = len(coords_np)
        self.area = window_geom.area
        if self.area <= 0:
            raise ValueError("Observation window area must be greater than zero.")

        self.mean = self.coords_np.mean(axis=0)
        self.std = self.coords_np.std(axis=0)
        self.std[self.std == 0] = 1.0
        self.quad_keep_mask = None
        self.quad_raw_np = None
        self.quad_counts_np = None
        self.quad_weights_np = None
        self.coords_obs = torch.tensor(
            (self.coords_np - self.mean) / self.std,
            dtype=torch.float32,
        )
        self.quad_coords, self.quad_counts, self.quad_weights = (
            self.make_berman_turner_grid(grid_size)
        )

    def make_berman_turner_grid(
        self, n_per_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_edges = np.linspace(self.x_min, self.x_max, n_per_dim + 1)
        y_edges = np.linspace(self.y_min, self.y_max, n_per_dim + 1)

        counts, _, _ = np.histogram2d(
            self.coords_np[:, 0],
            self.coords_np[:, 1],
            bins=[x_edges, y_edges],
        )

        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        gx, gy = np.meshgrid(x_centers, y_centers, indexing="ij")
        grid_raw = np.column_stack([gx.ravel(), gy.ravel()])
        grid_norm = (grid_raw - self.mean) / self.std

        y_counts_all = counts.ravel().astype(np.float64)
        grid_norm_all = grid_norm

        weights = []
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                cell = box(
                    x_edges[i],
                    y_edges[j],
                    x_edges[i + 1],
                    y_edges[j + 1],
                )
                weights.append(cell.intersection(self.window_geom).area)

        weights = np.asarray(weights, dtype=np.float64)
        keep = weights > 0
        if not np.any(keep):
            raise ValueError("No quadrature grid cells intersect the study window.")
        if n_per_dim == self.grid_size:
            self.quad_keep_mask = keep
            self.quad_raw_np = grid_raw[keep]
            self.quad_counts_np = y_counts_all[keep]
            self.quad_weights_np = weights[keep]

        grid_norm = grid_norm_all[keep]
        y_counts = y_counts_all[keep]
        weights = weights[keep]

        return (
            torch.tensor(grid_norm, dtype=torch.float32),
            torch.tensor(y_counts[:, None], dtype=torch.float32),
            torch.tensor(weights[:, None], dtype=torch.float32),
        )

    def bt_loglik_grid(
        self,
        model: nn.Module,
        q_coords: torch.Tensor,
        q_counts: torch.Tensor,
        q_weights: torch.Tensor,
    ) -> float:
        lambda_vals = model(q_coords)
        return (
            q_counts * torch.log(lambda_vals.clamp_min(MIN_LAMBDA))
            - q_weights * lambda_vals
        ).sum().item()

    def bt_nll_grid(
        self,
        model: nn.Module,
        q_coords: torch.Tensor,
        q_counts: torch.Tensor,
        q_weights: torch.Tensor,
    ) -> torch.Tensor:
        lambda_vals = model(q_coords)
        return -(
            q_counts * torch.log(lambda_vals.clamp_min(MIN_LAMBDA))
            - q_weights * lambda_vals
        ).sum()

    def bt_loglik(self, model: nn.Module) -> float:
        return self.bt_loglik_grid(
            model,
            self.quad_coords,
            self.quad_counts,
            self.quad_weights,
        )

    def bt_nll(self, model: nn.Module) -> torch.Tensor:
        return self.bt_nll_grid(
            model,
            self.quad_coords,
            self.quad_counts,
            self.quad_weights,
        )

    def fit(
        self,
        model: nn.Module,
        epochs: int,
        lr: float,
        q_coords: torch.Tensor | None = None,
        q_counts: torch.Tensor | None = None,
        q_weights: torch.Tensor | None = None,
    ) -> nn.Module:
        q_coords = self.quad_coords if q_coords is None else q_coords
        q_counts = self.quad_counts if q_counts is None else q_counts
        q_weights = self.quad_weights if q_weights is None else q_weights

        opt = optim.Adam(model.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = self.bt_nll_grid(model, q_coords, q_counts, q_weights)
            loss.backward()
            opt.step()
        return model

    def berman_turner_residuals(
        self, model: nn.Module
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            lambda_vals = model(self.quad_coords)
            expected_counts = self.quad_weights * lambda_vals
            raw_resid = self.quad_counts - expected_counts
            pearson_resid = raw_resid / torch.sqrt(
                expected_counts.clamp_min(MIN_EXPECTED_COUNT)
            )

        return (
            raw_resid.numpy().ravel(),
            pearson_resid.numpy().ravel(),
            expected_counts.numpy().ravel(),
            self.quad_counts.numpy().ravel(),
        )

    def predict_intensity_surface(
        self, model: nn.Module, n_per_dim: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.linspace(self.x_min, self.x_max, n_per_dim)
        ys = np.linspace(self.y_min, self.y_max, n_per_dim)
        gx, gy = np.meshgrid(xs, ys)

        grid_raw = np.column_stack([gx.ravel(), gy.ravel()])
        grid_norm = (grid_raw - self.mean) / self.std
        grid_tensor = torch.tensor(grid_norm, dtype=torch.float32)

        with torch.no_grad():
            intensity = model(grid_tensor).numpy().reshape(n_per_dim, n_per_dim)

        window_mask = np.array(
            [
                self.window_geom.intersects(Point(x, y))
                for x, y in grid_raw
            ],
            dtype=bool,
        ).reshape(n_per_dim, n_per_dim)
        intensity[~window_mask] = np.nan

        return gx, gy, intensity


def run_grid_sensitivity(
    experiment: WoodThrushExperiment,
    grid_sizes: list[int],
    lambda_hat: float,
    loglik_hppp: float,
    epochs_constant: int,
    epochs_linear: int,
    lr: float,
) -> pd.DataFrame:
    rows = []

    for grid_size in grid_sizes:
        q_coords, q_counts, q_weights = experiment.make_berman_turner_grid(grid_size)

        const_g = experiment.fit(
            ConstantIPPP(lambda_hat),
            epochs=epochs_constant,
            lr=lr,
            q_coords=q_coords,
            q_counts=q_counts,
            q_weights=q_weights,
        )
        ll_const = experiment.bt_loglik_grid(const_g, q_coords, q_counts, q_weights)

        lin_g = experiment.fit(
            LinearIPPP(input_dim=2, lambda_init=lambda_hat),
            epochs=epochs_linear,
            lr=lr,
            q_coords=q_coords,
            q_counts=q_counts,
            q_weights=q_weights,
        )
        ll_lin = experiment.bt_loglik_grid(lin_g, q_coords, q_counts, q_weights)

        lr_value = lr_stat(ll_lin, loglik_hppp)
        rows.append(
            {
                "grid_n_per_dim": grid_size,
                "n_cells": grid_size * grid_size,
                "HPPP_BT_LogLik": loglik_hppp,
                "Constant_BT_LogLik": ll_const,
                "Linear_BT_LogLik": ll_lin,
                "Linear_AIC": aic(ll_lin, 3),
                "Linear_BIC": bic(ll_lin, 3, experiment.n),
                "LR_Linear_vs_HPPP": lr_value,
                "LR_p_value": chi2.sf(lr_value, df=2),
                "beta_x": lin_g.linear.weight.detach().numpy()[0, 0],
                "beta_y": lin_g.linear.weight.detach().numpy()[0, 1],
                "beta_0": lin_g.linear.bias.item(),
            }
        )

    return pd.DataFrame(rows)


def assign_spatial_folds(
    coords_np: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    blocks_per_dim: int,
    folds: int,
) -> np.ndarray:
    if blocks_per_dim <= 0:
        raise ValueError("--cv-blocks-per-dim must be greater than zero.")
    if folds <= 1:
        raise ValueError("--cv-folds must be greater than one.")

    x_scaled = (coords_np[:, 0] - x_min) / max(x_max - x_min, 1e-12)
    y_scaled = (coords_np[:, 1] - y_min) / max(y_max - y_min, 1e-12)
    x_block = np.clip((x_scaled * blocks_per_dim).astype(int), 0, blocks_per_dim - 1)
    y_block = np.clip((y_scaled * blocks_per_dim).astype(int), 0, blocks_per_dim - 1)
    block_id = x_block * blocks_per_dim + y_block
    return block_id % folds


def point_loglik(model: nn.Module, coords_norm: torch.Tensor) -> float:
    with torch.no_grad():
        lambda_vals = model(coords_norm)
        return torch.log(lambda_vals.clamp_min(MIN_LAMBDA)).sum().item()


def integral_loglik_term(
    model: nn.Module,
    q_coords: torch.Tensor,
    q_weights: torch.Tensor,
) -> float:
    with torch.no_grad():
        lambda_vals = model(q_coords)
        return (q_weights * lambda_vals).sum().item()


def run_spatial_block_cv(
    experiment: WoodThrushExperiment,
    blocks_per_dim: int,
    folds: int,
    epochs_constant: int,
    epochs_linear: int,
    lr: float,
) -> pd.DataFrame:
    if experiment.quad_raw_np is None:
        raise ValueError("Main quadrature grid is unavailable for cross-validation.")

    point_folds = assign_spatial_folds(
        experiment.coords_np,
        experiment.x_min,
        experiment.x_max,
        experiment.y_min,
        experiment.y_max,
        blocks_per_dim,
        folds,
    )
    quad_folds = assign_spatial_folds(
        experiment.quad_raw_np,
        experiment.x_min,
        experiment.x_max,
        experiment.y_min,
        experiment.y_max,
        blocks_per_dim,
        folds,
    )

    rows = []
    for fold in range(folds):
        train_quad = quad_folds != fold
        test_quad = quad_folds == fold
        test_points = point_folds == fold

        train_count = float(experiment.quad_counts_np[train_quad].sum())
        train_area = float(experiment.quad_weights_np[train_quad].sum())
        test_count = int(test_points.sum())
        test_area = float(experiment.quad_weights_np[test_quad].sum())
        if train_count <= 0 or train_area <= 0 or test_area <= 0:
            continue

        lambda_train = train_count / train_area
        q_train_coords = experiment.quad_coords[train_quad]
        q_train_counts = experiment.quad_counts[train_quad]
        q_train_weights = experiment.quad_weights[train_quad]
        q_test_coords = experiment.quad_coords[test_quad]
        q_test_weights = experiment.quad_weights[test_quad]
        obs_test_coords = experiment.coords_obs[test_points]

        hppp_ll = test_count * np.log(lambda_train) - lambda_train * test_area

        const_model = experiment.fit(
            ConstantIPPP(lambda_train),
            epochs=epochs_constant,
            lr=lr,
            q_coords=q_train_coords,
            q_counts=q_train_counts,
            q_weights=q_train_weights,
        )
        const_ll = (
            point_loglik(const_model, obs_test_coords)
            - integral_loglik_term(const_model, q_test_coords, q_test_weights)
        )

        lin_model = experiment.fit(
            LinearIPPP(input_dim=2, lambda_init=lambda_train),
            epochs=epochs_linear,
            lr=lr,
            q_coords=q_train_coords,
            q_counts=q_train_counts,
            q_weights=q_train_weights,
        )
        lin_ll = (
            point_loglik(lin_model, obs_test_coords)
            - integral_loglik_term(lin_model, q_test_coords, q_test_weights)
        )

        rows.append(
            {
                "fold": fold,
                "test_observations": test_count,
                "test_area": test_area,
                "train_observations": train_count,
                "train_area": train_area,
                "HPPP_heldout_loglik": hppp_ll,
                "Constant_heldout_loglik": const_ll,
                "Linear_heldout_loglik": lin_ll,
            }
        )

    cv_df = pd.DataFrame(rows)
    if cv_df.empty:
        raise ValueError("Spatial block CV produced no valid folds.")
    return cv_df


def run_simulation_diagnostics(
    experiment: WoodThrushExperiment,
    model: nn.Module,
    n_simulations: int,
) -> pd.DataFrame:
    if n_simulations <= 0:
        raise ValueError("--simulation-count must be greater than zero.")

    with torch.no_grad():
        lambda_vals = model(experiment.quad_coords).numpy().ravel()
    expected = experiment.quad_weights_np * lambda_vals
    simulated_counts = np.random.poisson(expected[None, :], size=(n_simulations, len(expected)))
    simulated_totals = simulated_counts.sum(axis=1)
    observed_total = int(experiment.quad_counts_np.sum())

    return pd.DataFrame(
        {
            "simulation": np.arange(1, n_simulations + 1),
            "total_count": simulated_totals,
            "observed_total": observed_total,
            "expected_total": expected.sum(),
        }
    )


def inhomogeneous_k_diagnostic(
    experiment: WoodThrushExperiment,
    model: nn.Module,
    n_radii: int,
) -> pd.DataFrame:
    if n_radii <= 1:
        raise ValueError("--k-radii must be greater than one.")

    coords = experiment.coords_np
    with torch.no_grad():
        lambda_obs = model(experiment.coords_obs).numpy().ravel()

    dx = coords[:, 0][:, None] - coords[:, 0][None, :]
    dy = coords[:, 1][:, None] - coords[:, 1][None, :]
    distances = np.sqrt(dx * dx + dy * dy)
    not_self = ~np.eye(len(coords), dtype=bool)

    weights = np.zeros_like(distances, dtype=np.float64)
    lambda_product = lambda_obs[:, None] * lambda_obs[None, :]
    weights[not_self] = 1.0 / np.maximum(lambda_product[not_self], 1e-30)

    max_radius = 0.25 * min(experiment.x_max - experiment.x_min, experiment.y_max - experiment.y_min)
    radii = np.linspace(max_radius / n_radii, max_radius, n_radii)
    k_values = []
    for radius in radii:
        included = (distances <= radius) & not_self
        k_values.append(weights[included].sum() / experiment.area)

    return pd.DataFrame(
        {
            "radius": radii,
            "K_inhom": k_values,
            "Poisson_theoretical": np.pi * radii * radii,
        }
    )


def transform_xy(
    x_values: np.ndarray,
    y_values: np.ndarray,
    source_crs,
    target_crs,
) -> tuple[np.ndarray, np.ndarray]:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return transformer.transform(x_values, y_values)


def axis_labels(plot_crs) -> tuple[str, str]:
    crs = CRS.from_user_input(plot_crs)
    epsg = crs.to_epsg()
    if epsg == 4326:
        return "Longitude", "Latitude"
    name = crs.name or crs.to_string()
    return f"x ({name})", f"y ({name})"


def apply_map_axes(
    ax: Axes,
    plot_window: gpd.GeoSeries,
    plot_crs,
) -> None:
    x_label, y_label = axis_labels(plot_crs)
    bounds = plot_window.total_bounds
    x_pad = (bounds[2] - bounds[0]) * 0.03
    y_pad = (bounds[3] - bounds[1]) * 0.03
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(bounds[0] - x_pad, bounds[2] + x_pad)
    ax.set_ylim(bounds[1] - y_pad, bounds[3] + y_pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)


def draw_window_boundary(ax: Axes, plot_window: gpd.GeoSeries) -> None:
    plot_window.boundary.plot(ax=ax, color="black", linewidth=0.8, alpha=0.75, zorder=4)


def intensity_display_scale(analysis_crs) -> tuple[float, str]:
    crs = CRS.from_user_input(analysis_crs)
    unit_name = crs.axis_info[0].unit_name.lower() if crs.axis_info else ""
    if "metre" in unit_name or "meter" in unit_name:
        return 1e9, "Intensity (obs / 1,000 km2)"
    return 1e9, "Intensity (lambda x 1e9)"


def plot_point_pattern(
    experiment: WoodThrushExperiment,
    points: gpd.GeoDataFrame,
    plot_window: gpd.GeoSeries,
    plot_crs,
    image_dir: Path,
) -> None:
    plot_points = points.to_crs(plot_crs)
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_window_boundary(ax, plot_window)
    ax.scatter(
        plot_points.geometry.x,
        plot_points.geometry.y,
        s=8,
        color="darkgreen",
        alpha=0.35,
        label="Wood Thrush observations",
        zorder=5,
    )
    ax.set_title("Wood Thrush Observations")
    apply_map_axes(ax, plot_window, plot_crs)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(image_dir / "wood_thrush_point_pattern.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_intensity_comparison(
    experiment: WoodThrushExperiment,
    lin_model: LinearIPPP,
    lambda_hat: float,
    plot_grid_size: int,
    points: gpd.GeoDataFrame,
    analysis_crs,
    plot_window: gpd.GeoSeries,
    plot_crs,
    image_dir: Path,
) -> None:
    gx, gy, intensity = experiment.predict_intensity_surface(lin_model, plot_grid_size)
    gx_plot, gy_plot = transform_xy(gx, gy, analysis_crs, plot_crs)
    plot_points = points.to_crs(plot_crs)

    hppp_surface = np.full_like(intensity, lambda_hat)
    hppp_surface[np.isnan(intensity)] = np.nan

    scale, colorbar_label = intensity_display_scale(analysis_crs)
    intensity_plot = intensity * scale
    hppp_plot = hppp_surface * scale

    vmin = min(np.nanmin(hppp_plot), np.nanmin(intensity_plot))
    vmax = max(np.nanmax(hppp_plot), np.nanmax(intensity_plot))
    levels = np.linspace(vmin, vmax, 30)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes[0].contourf(gx_plot, gy_plot, hppp_plot, levels=levels, vmin=vmin, vmax=vmax)
    axes[0].scatter(plot_points.geometry.x, plot_points.geometry.y, s=6, color="black", alpha=0.25, zorder=5)
    axes[0].set_title("HPPP: Constant Intensity")
    apply_map_axes(axes[0], plot_window, plot_crs)

    im = axes[1].contourf(gx_plot, gy_plot, intensity_plot, levels=levels, vmin=vmin, vmax=vmax)
    axes[1].scatter(plot_points.geometry.x, plot_points.geometry.y, s=6, color="black", alpha=0.25, zorder=5)
    axes[1].set_title("Linear IPPP: Fitted Intensity")
    apply_map_axes(axes[1], plot_window, plot_crs)

    cbar = fig.colorbar(
        im,
        ax=axes,
        location="bottom",
        shrink=0.75,
        pad=0.08,
        aspect=40,
    )
    cbar.set_label(colorbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(image_dir / "wood_thrush_intensity_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residual_surface(
    experiment: WoodThrushExperiment,
    pearson_resid: np.ndarray,
    points: gpd.GeoDataFrame,
    analysis_crs,
    plot_window: gpd.GeoSeries,
    plot_crs,
    image_dir: Path,
) -> None:
    if experiment.quad_keep_mask is None:
        raise ValueError("Main quadrature mask is unavailable for residual plotting.")

    resid_grid = np.full(
        experiment.grid_size * experiment.grid_size,
        np.nan,
        dtype=np.float64,
    )
    resid_grid[experiment.quad_keep_mask] = pearson_resid
    resid_grid = resid_grid.reshape(experiment.grid_size, experiment.grid_size)

    x_edges = np.linspace(experiment.x_min, experiment.x_max, experiment.grid_size + 1)
    y_edges = np.linspace(experiment.y_min, experiment.y_max, experiment.grid_size + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    gx_resid, gy_resid = np.meshgrid(x_centers, y_centers, indexing="ij")
    gx_plot, gy_plot = transform_xy(gx_resid, gy_resid, analysis_crs, plot_crs)
    plot_points = points.to_crs(plot_crs)

    vmax = np.nanmax(np.abs(resid_grid))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    im = ax.contourf(
        gx_plot,
        gy_plot,
        resid_grid,
        levels=np.linspace(vmin, vmax, 31),
        vmin=vmin,
        vmax=vmax,
    )
    ax.scatter(plot_points.geometry.x, plot_points.geometry.y, s=5, color="black", alpha=0.2, zorder=5)
    ax.set_title("Pearson Residual Surface: Linear IPPP")
    apply_map_axes(ax, plot_window, plot_crs)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson residual", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(image_dir / "wood_thrush_linear_ippp_pearson_residuals.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grid_sensitivity(
    grid_sensitivity_df: pd.DataFrame,
    loglik_hppp: float,
    image_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        grid_sensitivity_df["grid_n_per_dim"],
        grid_sensitivity_df["Linear_BT_LogLik"],
        marker="o",
        label="Linear IPPP",
    )
    ax.axhline(loglik_hppp, linestyle="--", label="HPPP")
    ax.set_title("Sensitivity to Berman-Turner Grid Resolution")
    ax.set_xlabel("Grid cells per dimension")
    ax.set_ylabel("BT log-likelihood")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_dir / "wood_thrush_grid_sensitivity_loglik.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_spatial_block_cv(cv_df: pd.DataFrame, image_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cv_df["fold"], cv_df["HPPP_heldout_loglik"], marker="o", label="HPPP")
    ax.plot(cv_df["fold"], cv_df["Constant_heldout_loglik"], marker="o", label="Constant NN")
    ax.plot(cv_df["fold"], cv_df["Linear_heldout_loglik"], marker="o", label="Linear IPPP")
    ax.set_title("Spatial Block Cross-Validation")
    ax.set_xlabel("Held-out fold")
    ax.set_ylabel("Held-out IPPP log-likelihood")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_dir / "wood_thrush_spatial_block_cv.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_simulation_diagnostics(sim_df: pd.DataFrame, image_dir: Path) -> None:
    observed_total = float(sim_df["observed_total"].iloc[0])
    expected_total = float(sim_df["expected_total"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(sim_df["total_count"], bins=30, color="steelblue", alpha=0.75)
    ax.axvline(observed_total, color="black", linewidth=2, label="Observed")
    ax.axvline(expected_total, color="darkorange", linestyle="--", linewidth=2, label="Expected")
    ax.set_title("Simulation Diagnostic: Total Count")
    ax.set_xlabel("Simulated total observations")
    ax.set_ylabel("Simulation frequency")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_dir / "wood_thrush_simulated_total_count.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_inhomogeneous_k(k_df: pd.DataFrame, image_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k_df["radius"], k_df["K_inhom"], label="Estimated Kinhom")
    ax.plot(k_df["radius"], k_df["Poisson_theoretical"], linestyle="--", label="Poisson")
    ax.set_title("Approximate Inhomogeneous K-Function")
    ax.set_xlabel("Radius")
    ax.set_ylabel("K(r)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_dir / "wood_thrush_inhomogeneous_k.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    image_dir = Path(args.image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    points = load_points(Path(args.input))
    if args.analysis_crs:
        points = points.to_crs(args.analysis_crs)

    bbox = validate_bbox(args)
    window_geom = build_window_geometry(points, args.boundary, bbox)
    points = filter_points_to_window(
        points,
        window_geom,
        should_filter=args.boundary is not None or bbox is not None,
    )

    coords_np = np.column_stack([points.geometry.x, points.geometry.y]).astype(np.float64)

    experiment = WoodThrushExperiment(
        coords_np=coords_np,
        window_geom=window_geom,
        grid_size=args.grid_size,
    )
    analysis_crs = points.crs
    plot_crs = CRS.from_user_input(args.plot_crs)
    plot_window = gpd.GeoSeries([window_geom], crs=analysis_crs).to_crs(plot_crs)

    lambda_hat = experiment.n / experiment.area
    loglik_hppp = experiment.n * np.log(lambda_hat) - lambda_hat * experiment.area

    const_model = experiment.fit(
        ConstantIPPP(lambda_hat),
        epochs=args.epochs_constant,
        lr=args.lr,
    )
    loglik_const = experiment.bt_loglik(const_model)

    lin_model = experiment.fit(
        LinearIPPP(input_dim=2, lambda_init=lambda_hat),
        epochs=args.epochs_linear,
        lr=args.lr,
    )
    loglik_lin = experiment.bt_loglik(lin_model)

    results_df = pd.DataFrame(
        {
            "Model": ["HPPP", "Constant NN", "Linear IPPP"],
            "k": [1, 1, 3],
            "BT_LogLik": [loglik_hppp, loglik_const, loglik_lin],
            "AIC": [aic(loglik_hppp, 1), aic(loglik_const, 1), aic(loglik_lin, 3)],
            "BIC": [
                bic(loglik_hppp, 1, experiment.n),
                bic(loglik_const, 1, experiment.n),
                bic(loglik_lin, 3, experiment.n),
            ],
        }
    )
    print(results_df)

    lr_lin_vs_hppp = lr_stat(loglik_lin, loglik_hppp)
    p_lin_vs_hppp = chi2.sf(lr_lin_vs_hppp, df=2)

    print("\nLikelihood Ratio Test:")
    print(f"Linear vs HPPP LR stat: {lr_lin_vs_hppp:.4f}")
    print(f"Linear vs HPPP p-value: {p_lin_vs_hppp:.6g}")

    print("\nSanity checks:")
    print(f"Observations: {experiment.n:,}")
    print(f"Window area: {experiment.area:.4f}")
    print(f"lambda_hat: {lambda_hat:.10f}")
    print(f"HPPP loglik: {loglik_hppp:.4f}")
    print(f"Constant NN loglik: {loglik_const:.4f}")
    print(f"Linear IPPP loglik: {loglik_lin:.4f}")

    print("\nCoefficients:")
    weight = lin_model.linear.weight.detach().numpy()
    print(f"Linear weight: {weight}")
    print(f"Linear bias: {lin_model.linear.bias.item()}")
    print(
        "Fitted model is approximately: "
        f"lambda(s) = exp({lin_model.linear.bias.item():.4f} "
        f"+ {weight[0, 0]:.4f}*x + {weight[0, 1]:.4f}*y)"
    )

    raw_resid, pearson_resid, expected_counts, observed_counts = (
        experiment.berman_turner_residuals(lin_model)
    )
    print("\nSpatial Residual Diagnostics:")
    print(f"Mean raw residual: {raw_resid.mean():.6f}")
    print(f"Mean Pearson residual: {pearson_resid.mean():.6f}")
    print(f"Pearson residual SD: {pearson_resid.std():.6f}")
    print(f"Observed total count: {observed_counts.sum():.0f}")
    print(f"Expected total count: {expected_counts.sum():.4f}")

    grid_sensitivity_df = run_grid_sensitivity(
        experiment=experiment,
        grid_sizes=args.grid_sensitivity,
        lambda_hat=lambda_hat,
        loglik_hppp=loglik_hppp,
        epochs_constant=args.epochs_constant,
        epochs_linear=args.epochs_linear,
        lr=args.lr,
    )
    print("\nBerman-Turner Grid Sensitivity:")
    print(grid_sensitivity_df)

    cv_df = run_spatial_block_cv(
        experiment=experiment,
        blocks_per_dim=args.cv_blocks_per_dim,
        folds=args.cv_folds,
        epochs_constant=args.epochs_constant,
        epochs_linear=args.epochs_linear,
        lr=args.lr,
    )
    print("\nSpatial Block Cross-Validation:")
    print(cv_df)
    print("\nSpatial Block CV totals:")
    print(
        cv_df[
            [
                "HPPP_heldout_loglik",
                "Constant_heldout_loglik",
                "Linear_heldout_loglik",
            ]
        ].sum()
    )

    simulation_df = run_simulation_diagnostics(
        experiment=experiment,
        model=lin_model,
        n_simulations=args.simulation_count,
    )
    observed_total = float(simulation_df["observed_total"].iloc[0])
    simulated_totals = simulation_df["total_count"].to_numpy()
    lower_tail = np.mean(simulated_totals <= observed_total)
    upper_tail = np.mean(simulated_totals >= observed_total)
    print("\nSimulation Diagnostics:")
    print(f"Observed total count: {observed_total:.0f}")
    print(f"Simulated total mean: {simulated_totals.mean():.2f}")
    print(f"Simulated total 2.5%-97.5%: {np.quantile(simulated_totals, [0.025, 0.975])}")
    print(f"Two-sided simulation p-value: {2 * min(lower_tail, upper_tail):.4f}")

    k_df = inhomogeneous_k_diagnostic(
        experiment=experiment,
        model=lin_model,
        n_radii=args.k_radii,
    )
    print("\nInhomogeneous K Diagnostic:")
    print(k_df.head())

    cv_df.to_csv(image_dir / "wood_thrush_spatial_block_cv.csv", index=False)
    simulation_df.to_csv(image_dir / "wood_thrush_simulation_diagnostics.csv", index=False)
    k_df.to_csv(image_dir / "wood_thrush_inhomogeneous_k.csv", index=False)

    plot_point_pattern(
        experiment,
        points,
        plot_window,
        plot_crs,
        image_dir,
    )
    plot_intensity_comparison(
        experiment,
        lin_model,
        lambda_hat,
        args.plot_grid_size,
        points,
        analysis_crs,
        plot_window,
        plot_crs,
        image_dir,
    )
    plot_residual_surface(
        experiment,
        pearson_resid,
        points,
        analysis_crs,
        plot_window,
        plot_crs,
        image_dir,
    )
    plot_grid_sensitivity(grid_sensitivity_df, loglik_hppp, image_dir)
    plot_spatial_block_cv(cv_df, image_dir)
    plot_simulation_diagnostics(simulation_df, image_dir)
    plot_inhomogeneous_k(k_df, image_dir)

    print(f"\nSaved figures to {image_dir}")


if __name__ == "__main__":
    main()

# TODO:
# - Adding a nonlinear neural network intensity model
# - Comparing linear and nonlinear IPPPs with AIC, BIC, and held-out likelihood
# - Extending the mark model to nonlinear conditional means or heteroskedastic variance
# - Incorporating spatial covariates
# - Adding temporal terms and diagnostics
