"""
Spatial IPPP experiment for Wood Thrush eBird observations.

This mirrors and extends the spatial portion of exp/nippp.py:

- HPPP closed-form baseline
- Constant neural IPPP sanity check
- Linear IPPP with x/y trend
- Regularized nonlinear neural IPPP with x/y trend
- Optional cyclic day-of-year temporal terms
- Optional raster-backed spatial covariates
- likelihood-ratio test
- Berman-Turner residual diagnostics
- spatial block cross-validation
- simulation-based total-count diagnostics
- approximate inhomogeneous K diagnostics
- grid-resolution sensitivity

Temporal terms and spatial covariates can be toggled independently or used
together in the same model. When both are enabled, the linear and nonlinear
IPPPs use x/y coordinates, selected raster covariates, and cyclic day-of-year
features, and all diagnostics are recomputed for that combined feature set.
The observation window can be an irregular boundary dataset, a WGS84 bbox, or
the rectangular extent of the observations.

Run from the project root:

    python exp/wood_thrush_nippp.py --input data/wood_thrush_nc_2020_2023.geojson

Combined temporal plus covariate run:

    python exp/wood_thrush_nippp.py --input data/wood_thrush_nc_2020_2023_covariates.geojson --boundary data/boundaries/nc_state_boundary.gpkg --analysis-crs EPSG:5070 --plot-crs EPSG:4326 --image-dir images/wood_thrush_nippp_temporal_covariates --covariate-raster data/nc_covariate_stack.tif --covariates canopy_median nc_usgs30m_match_tcc distance_to_waterbody_m distance_to_coastline_m --cv-blocks-per-dim 5 --cv-folds 5 --simulation-count 500 --k-radii 50 --epochs-nonlinear 10000 --hidden-dim 16 --hidden-layers 1 --dropout 0.10 --nonlinear-lr 5e-4 --nonlinear-weight-decay 1e-3 --temporal-bins 12 --plot-day-of-year 150
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
import rasterio
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
DEFAULT_COVARIATES = [
    "canopy_median",
    "nc_usgs30m_match_tcc",
    "distance_to_waterbody_m",
    "distance_to_coastline_m",
]


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
        "--epochs-nonlinear",
        type=int,
        default=10000,
        help="Training epochs for the nonlinear neural IPPP. Defaults to 10000.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=16,
        help="Hidden layer width for the nonlinear neural IPPP. Defaults to 16.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=1,
        help="Number of hidden layers for the nonlinear neural IPPP. Defaults to 1.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help="Dropout rate for nonlinear hidden layers. Defaults to 0.10.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate. Defaults to 1e-3.",
    )
    parser.add_argument(
        "--nonlinear-lr",
        type=float,
        default=5e-4,
        help="Adam learning rate for the nonlinear IPPP. Defaults to 5e-4.",
    )
    parser.add_argument(
        "--nonlinear-weight-decay",
        type=float,
        default=1e-3,
        help="Adam weight decay for the nonlinear IPPP. Defaults to 1e-3.",
    )
    parser.add_argument(
        "--intensity-plot-scale",
        choices=("relative", "absolute"),
        default="relative",
        help="Scale for intensity comparison plots. Defaults to relative.",
    )
    parser.add_argument(
        "--date-column",
        default="queryDate",
        help="Date column used for cyclic temporal terms. Defaults to queryDate.",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_false",
        dest="use_temporal",
        help="Disable cyclic day-of-year terms and fit a spatial-only model.",
    )
    parser.set_defaults(use_temporal=True)
    parser.add_argument(
        "--temporal-bins",
        type=int,
        default=12,
        help="Number of annual-cycle bins in the space-time quadrature. Defaults to 12.",
    )
    parser.add_argument(
        "--plot-day-of-year",
        type=int,
        default=150,
        help="Day of year used for map intensity surfaces when temporal terms are enabled. Defaults to 150.",
    )
    parser.add_argument(
        "--covariate-raster",
        help=(
            "Optional raster stack used to sample spatial covariates at observed "
            "points and quadrature cells."
        ),
    )
    parser.add_argument(
        "--covariates",
        nargs="+",
        default=None,
        help=(
            "Covariate columns to use from --covariate-raster. Defaults to "
            "canopy_median nc_usgs30m_match_tcc distance_to_waterbody_m "
            "distance_to_coastline_m when a covariate raster is provided."
        ),
    )
    parser.add_argument(
        "--canopy-prefix",
        default="tcc_",
        help="Band-name prefix used to identify yearly canopy bands. Defaults to tcc_.",
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


def temporal_phase_from_dates(
    points: gpd.GeoDataFrame,
    date_column: str,
) -> tuple[np.ndarray, float]:
    if date_column not in points.columns:
        raise ValueError(f"Date column is not in the input dataset: {date_column}")

    dates = pd.to_datetime(points[date_column], errors="coerce")
    if dates.isna().any():
        missing = int(dates.isna().sum())
        raise ValueError(f"Date column {date_column} has {missing:,} missing or invalid values.")

    years = dates.dt.year
    year_lengths = np.where(dates.dt.is_leap_year, 366.0, 365.0)
    phase = ((dates.dt.dayofyear.to_numpy(dtype=np.float64) - 1.0) / year_lengths) % 1.0

    unique_years = np.sort(years.unique())
    temporal_duration = float(
        sum(366.0 if pd.Timestamp(year=int(year), month=12, day=31).is_leap_year else 365.0 for year in unique_years)
    )
    return phase, temporal_duration


def phase_from_day_of_year(day_of_year: int) -> float:
    if day_of_year < 1 or day_of_year > 366:
        raise ValueError("--plot-day-of-year must be between 1 and 366.")
    return ((float(day_of_year) - 1.0) / 365.25) % 1.0


def selected_covariates(args: argparse.Namespace) -> list[str]:
    if args.covariate_raster is None:
        if args.covariates is not None:
            raise ValueError("--covariates requires --covariate-raster.")
        return []
    return args.covariates if args.covariates is not None else list(DEFAULT_COVARIATES)


def snake_case(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "band"


def unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    unique = []
    for name in names:
        count = seen.get(name, 0)
        seen[name] = count + 1
        unique.append(name if count == 0 else f"{name}_{count + 1}")
    return unique


class RasterCovariateSampler:
    def __init__(
        self,
        raster_path: Path,
        selected_covariates: list[str],
        canopy_prefix: str,
    ):
        if not raster_path.exists():
            raise FileNotFoundError(f"Covariate raster does not exist: {raster_path}")

        self.raster_path = raster_path
        self.selected_covariates = selected_covariates
        self.canopy_prefix = canopy_prefix
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise ValueError(f"Covariate raster has no CRS: {raster_path}")
            self.crs = src.crs
            self.band_names = self._band_names(src)
            self.nodatavals = src.nodatavals

        missing = [name for name in selected_covariates if name not in self.available_covariates]
        if missing:
            raise ValueError(
                "Requested covariate(s) are unavailable in the raster stack: "
                + ", ".join(missing)
                + f". Available: {', '.join(self.available_covariates)}"
            )

    @staticmethod
    def _band_names(src: rasterio.DatasetReader) -> list[str]:
        names = []
        for index, description in enumerate(src.descriptions, start=1):
            names.append(snake_case(description if description else f"band_{index}"))
        return unique_names(names)

    @property
    def canopy_columns(self) -> list[str]:
        return [name for name in self.band_names if name.startswith(self.canopy_prefix)]

    @property
    def available_covariates(self) -> list[str]:
        names = list(self.band_names)
        if self.canopy_columns:
            names.append("canopy_median")
        return names

    def selected_values_from_bands(self, band_values: np.ndarray) -> np.ndarray:
        values_by_name = {
            name: band_values[:, band_index]
            for band_index, name in enumerate(self.band_names)
        }
        if self.canopy_columns:
            canopy_values = np.column_stack([values_by_name[name] for name in self.canopy_columns])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                values_by_name["canopy_median"] = np.nanmedian(canopy_values, axis=1)
        return np.column_stack([values_by_name[name] for name in self.selected_covariates])

    def clean_nodata(self, values: np.ndarray, src: rasterio.DatasetReader) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        for band_index, nodata in enumerate(src.nodatavals):
            if nodata is not None:
                values[:, band_index] = np.where(
                    values[:, band_index] == nodata,
                    np.nan,
                    values[:, band_index],
                )
        return values

    def nearest_valid_covariates(
        self,
        src: rasterio.DatasetReader,
        x: float,
        y: float,
        max_radius_pixels: int = 8,
    ) -> np.ndarray | None:
        row, col = src.index(x, y)
        row = int(np.clip(row, 0, src.height - 1))
        col = int(np.clip(col, 0, src.width - 1))
        best_distance = np.inf
        best_values = None

        for radius in range(1, max_radius_pixels + 1):
            row_start = max(0, row - radius)
            row_stop = min(src.height, row + radius + 1)
            col_start = max(0, col - radius)
            col_stop = min(src.width, col + radius + 1)
            if row_stop <= row_start or col_stop <= col_start:
                continue
            window = rasterio.windows.Window(
                col_start,
                row_start,
                col_stop - col_start,
                row_stop - row_start,
            )
            data = src.read(window=window, masked=True).astype(np.float64).filled(np.nan)
            band_pixels = data.reshape(src.count, -1).T
            band_pixels = self.clean_nodata(band_pixels, src)
            selected = self.selected_values_from_bands(band_pixels)
            valid = ~np.isnan(selected).any(axis=1)
            if not np.any(valid):
                continue

            rows, cols = np.indices((row_stop - row_start, col_stop - col_start))
            rows = rows.ravel() + row_start
            cols = cols.ravel() + col_start
            distances = (rows - row) ** 2 + (cols - col) ** 2
            valid_indices = np.flatnonzero(valid)
            nearest_index = valid_indices[np.argmin(distances[valid_indices])]
            nearest_distance = distances[nearest_index]
            if nearest_distance < best_distance:
                best_distance = nearest_distance
                best_values = selected[nearest_index]
            break

        return best_values

    def sample(self, spatial_raw: np.ndarray, source_crs) -> np.ndarray:
        return self.sample_with_mask(spatial_raw, source_crs, allow_missing=False)[0]

    def sample_with_mask(
        self,
        spatial_raw: np.ndarray,
        source_crs,
        allow_missing: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        transformer = Transformer.from_crs(source_crs, self.crs, always_xy=True)
        xs, ys = transformer.transform(spatial_raw[:, 0], spatial_raw[:, 1])
        coordinates = list(zip(xs, ys))

        with rasterio.open(self.raster_path) as src:
            sampled = np.ma.asarray(list(src.sample(coordinates, masked=True)), dtype=np.float64)
            sampled = sampled.filled(np.nan)
            sampled = self.clean_nodata(sampled, src)
            output = self.selected_values_from_bands(sampled)

            missing_rows = np.flatnonzero(np.isnan(output).any(axis=1))
            for row_index in missing_rows:
                replacement = self.nearest_valid_covariates(
                    src,
                    coordinates[row_index][0],
                    coordinates[row_index][1],
                )
                if replacement is not None:
                    output[row_index] = replacement

        missing_mask = np.isnan(output).any(axis=1)
        if missing_mask.any() and not allow_missing:
            bad_rows = int(np.isnan(output).any(axis=1).sum())
            raise ValueError(
                f"Sampled covariates contain missing values for {bad_rows:,} row(s). "
                "Check raster coverage, boundary, and selected covariates."
            )
        return output.astype(np.float64), ~missing_mask


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
        eta = torch.clamp(eta, min=-50, max=20)
        return torch.exp(eta)


class NonlinearIPPP(nn.Module):
    """Neural spatial IPPP: lambda(s) = exp(f_theta(s))."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        hidden_layers: int,
        lambda_init: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("--hidden-dim must be greater than zero.")
        if hidden_layers <= 0:
            raise ValueError("--hidden-layers must be greater than zero.")
        if not 0 <= dropout < 1:
            raise ValueError("--dropout must be in [0, 1).")

        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        output = nn.Linear(current_dim, 1)
        layers.append(output)
        self.network = nn.Sequential(*layers)

        with torch.no_grad():
            output.weight.zero_()
            output.bias.fill_(np.log(lambda_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eta = self.network(x)
        eta = torch.clamp(eta, min=-50, max=20)
        return torch.exp(eta)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def aic(ll: float, k: int) -> float:
    return 2 * k - 2 * ll


def bic(ll: float, k: int, n: int) -> float:
    return k * np.log(n) - 2 * ll


def lr_stat(ll_full: float, ll_reduced: float) -> float:
    return 2 * (ll_full - ll_reduced)


def two_sided_simulation_pvalue(values: np.ndarray, observed: float) -> float:
    lower_tail = np.mean(values <= observed)
    upper_tail = np.mean(values >= observed)
    return min(1.0, 2.0 * min(lower_tail, upper_tail))


class WoodThrushExperiment:
    def __init__(
        self,
        coords_np: np.ndarray,
        window_geom,
        grid_size: int,
        coord_crs,
        covariate_sampler: RasterCovariateSampler | None = None,
        temporal_phase_np: np.ndarray | None = None,
        temporal_duration: float = 1.0,
        temporal_bins: int = 1,
    ):
        self.coords_np = coords_np.astype(np.float64)
        self.coord_crs = coord_crs
        self.covariate_sampler = covariate_sampler
        self.use_covariates = covariate_sampler is not None
        self.covariate_names = (
            covariate_sampler.selected_covariates if covariate_sampler is not None else []
        )
        self.temporal_phase_np = temporal_phase_np
        self.use_temporal = temporal_phase_np is not None
        if self.use_temporal and len(temporal_phase_np) != len(coords_np):
            raise ValueError("Temporal phase array must have one value per observation.")
        if temporal_duration <= 0:
            raise ValueError("Temporal duration must be greater than zero.")
        if temporal_bins <= 0:
            raise ValueError("--temporal-bins must be greater than zero.")
        self.temporal_duration = float(temporal_duration if self.use_temporal else 1.0)
        self.temporal_bins = int(temporal_bins if self.use_temporal else 1)
        self.window_geom = window_geom
        self.x_min, self.y_min, self.x_max, self.y_max = window_geom.bounds
        self.grid_size = grid_size
        self.n = len(coords_np)
        self.area = window_geom.area
        self.measure = self.area * self.temporal_duration
        if self.area <= 0:
            raise ValueError("Observation window area must be greater than zero.")

        self.mean = self.coords_np.mean(axis=0)
        self.std = self.coords_np.std(axis=0)
        self.std[self.std == 0] = 1.0
        if self.use_covariates:
            self.covariates_np = self.covariate_sampler.sample(self.coords_np, self.coord_crs)
            self.covariate_mean = self.covariates_np.mean(axis=0)
            self.covariate_std = self.covariates_np.std(axis=0)
            self.covariate_std[self.covariate_std == 0] = 1.0
        else:
            self.covariates_np = None
            self.covariate_mean = None
            self.covariate_std = None
        self.quad_keep_mask = None
        self.quad_raw_np = None
        self.quad_spatial_raw_np = None
        self.quad_spatial_weights_np = None
        self.quad_spatial_index_np = None
        self.quad_temporal_phase_np = None
        self.quad_covariates_np = None
        self.quad_counts_np = None
        self.quad_weights_np = None
        self.coords_obs = torch.tensor(
            self.make_features(
                self.coords_np,
                temporal_phase=self.temporal_phase_np,
                covariates_raw=self.covariates_np,
            ),
            dtype=torch.float32,
        )
        self.quad_coords, self.quad_counts, self.quad_weights = (
            self.make_berman_turner_grid(grid_size)
        )

    @property
    def input_dim(self) -> int:
        dim = 2
        if self.use_covariates:
            dim += len(self.covariate_names)
        if self.use_temporal:
            dim += 2
        return dim

    def make_features(
        self,
        spatial_raw: np.ndarray,
        temporal_phase: np.ndarray | None = None,
        covariates_raw: np.ndarray | None = None,
    ) -> np.ndarray:
        parts = [(spatial_raw - self.mean) / self.std]

        if self.use_covariates:
            if covariates_raw is None:
                covariates_raw = self.covariate_sampler.sample(spatial_raw, self.coord_crs)
            covariates = (covariates_raw - self.covariate_mean) / self.covariate_std
            parts.append(covariates)

        if not self.use_temporal:
            return np.column_stack(parts)

        if temporal_phase is None:
            raise ValueError("Temporal phase is required for temporal models.")
        phase = np.asarray(temporal_phase, dtype=np.float64).reshape(-1)
        if len(phase) != len(spatial_raw):
            raise ValueError("Temporal phase must have one value per feature row.")
        angle = 2.0 * np.pi * phase
        temporal = np.column_stack([np.sin(angle), np.cos(angle)])
        parts.append(temporal)
        return np.column_stack(parts)

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
        y_counts_all = counts.ravel().astype(np.float64)

        weights = []
        sample_points = []
        for i in range(n_per_dim):
            for j in range(n_per_dim):
                cell = box(
                    x_edges[i],
                    y_edges[j],
                    x_edges[i + 1],
                    y_edges[j + 1],
                )
                intersection = cell.intersection(self.window_geom)
                weights.append(intersection.area)
                if intersection.is_empty:
                    sample_points.append((gx[i, j], gy[i, j]))
                else:
                    point = intersection.representative_point()
                    sample_points.append((point.x, point.y))

        weights = np.asarray(weights, dtype=np.float64)
        sample_points = np.asarray(sample_points, dtype=np.float64)
        keep = weights > 0
        if not np.any(keep):
            raise ValueError("No quadrature grid cells intersect the study window.")

        spatial_raw = sample_points[keep]
        spatial_weights = weights[keep]
        spatial_covariates = (
            self.covariate_sampler.sample(spatial_raw, self.coord_crs)
            if self.use_covariates
            else None
        )

        if self.use_temporal:
            t_edges = np.linspace(0.0, 1.0, self.temporal_bins + 1)
            t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])
            counts_3d, _ = np.histogramdd(
                np.column_stack(
                    [
                        self.coords_np[:, 0],
                        self.coords_np[:, 1],
                        self.temporal_phase_np,
                    ]
                ),
                bins=[x_edges, y_edges, t_edges],
            )
            counts_spacetime = counts_3d.reshape(n_per_dim * n_per_dim, self.temporal_bins)
            counts_spacetime = counts_spacetime[keep, :]
            spatial_index = np.repeat(np.arange(len(spatial_raw)), self.temporal_bins)
            temporal_phase = np.tile(t_centers, len(spatial_raw))
            feature_spatial = np.repeat(spatial_raw, self.temporal_bins, axis=0)
            feature_covariates = (
                np.repeat(spatial_covariates, self.temporal_bins, axis=0)
                if self.use_covariates
                else None
            )
            feature_matrix = self.make_features(
                feature_spatial,
                temporal_phase=temporal_phase,
                covariates_raw=feature_covariates,
            )
            y_counts = counts_spacetime.ravel().astype(np.float64)
            temporal_width = self.temporal_duration / self.temporal_bins
            weights = np.repeat(spatial_weights, self.temporal_bins) * temporal_width
            quad_raw = feature_spatial
            quad_covariates = feature_covariates
        else:
            spatial_index = np.arange(len(spatial_raw))
            temporal_phase = None
            feature_matrix = self.make_features(spatial_raw, covariates_raw=spatial_covariates)
            y_counts = y_counts_all[keep]
            weights = spatial_weights
            quad_raw = spatial_raw
            quad_covariates = spatial_covariates

        if n_per_dim == self.grid_size:
            self.quad_keep_mask = keep
            self.quad_raw_np = quad_raw
            self.quad_spatial_raw_np = spatial_raw
            self.quad_spatial_weights_np = spatial_weights
            self.quad_spatial_index_np = spatial_index
            self.quad_temporal_phase_np = temporal_phase
            self.quad_covariates_np = quad_covariates
            self.quad_counts_np = y_counts
            self.quad_weights_np = weights

        return (
            torch.tensor(feature_matrix, dtype=torch.float32),
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
        weight_decay: float = 0.0,
        q_coords: torch.Tensor | None = None,
        q_counts: torch.Tensor | None = None,
        q_weights: torch.Tensor | None = None,
    ) -> nn.Module:
        q_coords = self.quad_coords if q_coords is None else q_coords
        q_counts = self.quad_counts if q_counts is None else q_counts
        q_weights = self.quad_weights if q_weights is None else q_weights

        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            loss = self.bt_nll_grid(model, q_coords, q_counts, q_weights)
            loss.backward()
            opt.step()
        model.eval()
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
        self,
        model: nn.Module,
        n_per_dim: int,
        temporal_phase: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.linspace(self.x_min, self.x_max, n_per_dim)
        ys = np.linspace(self.y_min, self.y_max, n_per_dim)
        gx, gy = np.meshgrid(xs, ys)

        grid_raw = np.column_stack([gx.ravel(), gy.ravel()])
        window_mask = np.array(
            [
                self.window_geom.intersects(Point(x, y))
                for x, y in grid_raw
            ],
            dtype=bool,
        )
        grid_features = np.zeros((len(grid_raw), self.input_dim), dtype=np.float64)
        if self.use_temporal:
            if temporal_phase is None:
                raise ValueError("A temporal phase is required for temporal intensity prediction.")
            valid_grid = window_mask.copy()
            if self.use_covariates:
                grid_covariates, valid_covariates = self.covariate_sampler.sample_with_mask(
                    grid_raw[window_mask],
                    self.coord_crs,
                    allow_missing=True,
                )
                valid_indices = np.flatnonzero(window_mask)
                valid_grid[valid_indices[~valid_covariates]] = False
                grid_covariates = grid_covariates[valid_covariates]
            else:
                grid_covariates = None
            phase = np.full(valid_grid.sum(), temporal_phase, dtype=np.float64)
            grid_features[valid_grid] = self.make_features(
                grid_raw[valid_grid],
                temporal_phase=phase,
                covariates_raw=grid_covariates,
            )
        else:
            valid_grid = window_mask.copy()
            if self.use_covariates:
                grid_covariates, valid_covariates = self.covariate_sampler.sample_with_mask(
                    grid_raw[window_mask],
                    self.coord_crs,
                    allow_missing=True,
                )
                valid_indices = np.flatnonzero(window_mask)
                valid_grid[valid_indices[~valid_covariates]] = False
                grid_covariates = grid_covariates[valid_covariates]
            else:
                grid_covariates = None
            grid_features[valid_grid] = self.make_features(
                grid_raw[valid_grid],
                covariates_raw=grid_covariates,
            )
        grid_tensor = torch.tensor(grid_features, dtype=torch.float32)

        with torch.no_grad():
            intensity = model(grid_tensor).numpy().reshape(n_per_dim, n_per_dim)

        intensity[~valid_grid.reshape(n_per_dim, n_per_dim)] = np.nan

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
            LinearIPPP(input_dim=experiment.input_dim, lambda_init=lambda_hat),
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
                "Linear_AIC": aic(ll_lin, experiment.input_dim + 1),
                "Linear_BIC": bic(ll_lin, experiment.input_dim + 1, experiment.n),
                "LR_Linear_vs_HPPP": lr_value,
                "LR_p_value": chi2.sf(lr_value, df=experiment.input_dim),
                "beta_x": lin_g.linear.weight.detach().numpy()[0, 0],
                "beta_y": lin_g.linear.weight.detach().numpy()[0, 1],
                **{
                    f"beta_{name}": lin_g.linear.weight.detach().numpy()[0, index + 2]
                    for index, name in enumerate(experiment.covariate_names)
                },
                "beta_sin_doy": (
                    lin_g.linear.weight.detach().numpy()[0, 2 + len(experiment.covariate_names)]
                    if experiment.use_temporal
                    else np.nan
                ),
                "beta_cos_doy": (
                    lin_g.linear.weight.detach().numpy()[0, 3 + len(experiment.covariate_names)]
                    if experiment.use_temporal
                    else np.nan
                ),
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
    epochs_nonlinear: int,
    hidden_dim: int,
    hidden_layers: int,
    dropout: float,
    lr: float,
    nonlinear_lr: float,
    nonlinear_weight_decay: float,
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
            LinearIPPP(input_dim=experiment.input_dim, lambda_init=lambda_train),
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

        nonlinear_model = experiment.fit(
            NonlinearIPPP(
                input_dim=experiment.input_dim,
                hidden_dim=hidden_dim,
                hidden_layers=hidden_layers,
                lambda_init=lambda_train,
                dropout=dropout,
            ),
            epochs=epochs_nonlinear,
            lr=nonlinear_lr,
            weight_decay=nonlinear_weight_decay,
            q_coords=q_train_coords,
            q_counts=q_train_counts,
            q_weights=q_train_weights,
        )
        nonlinear_ll = (
            point_loglik(nonlinear_model, obs_test_coords)
            - integral_loglik_term(nonlinear_model, q_test_coords, q_test_weights)
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
                "Nonlinear_heldout_loglik": nonlinear_ll,
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
    if experiment.use_temporal:
        t_centers = np.linspace(
            0.5 / experiment.temporal_bins,
            1.0 - 0.5 / experiment.temporal_bins,
            experiment.temporal_bins,
        )
        temporal_width = experiment.temporal_duration / experiment.temporal_bins
        integrated = np.zeros(len(coords), dtype=np.float64)
        for phase in t_centers:
            phases = np.full(len(coords), phase, dtype=np.float64)
            features = experiment.make_features(
                coords,
                temporal_phase=phases,
                covariates_raw=experiment.covariates_np,
            )
            with torch.no_grad():
                lambda_vals = model(torch.tensor(features, dtype=torch.float32)).numpy().ravel()
            integrated += lambda_vals * temporal_width
        lambda_obs = integrated
    else:
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
    nonlinear_model: NonlinearIPPP,
    lambda_hat: float,
    plot_grid_size: int,
    plot_temporal_phase: float | None,
    plot_day_of_year: int,
    points: gpd.GeoDataFrame,
    analysis_crs,
    plot_window: gpd.GeoSeries,
    plot_crs,
    intensity_plot_scale: str,
    image_dir: Path,
) -> None:
    gx, gy, intensity = experiment.predict_intensity_surface(
        lin_model,
        plot_grid_size,
        temporal_phase=plot_temporal_phase,
    )
    _, _, nonlinear_intensity = experiment.predict_intensity_surface(
        nonlinear_model,
        plot_grid_size,
        temporal_phase=plot_temporal_phase,
    )
    gx_plot, gy_plot = transform_xy(gx, gy, analysis_crs, plot_crs)
    plot_points = points.to_crs(plot_crs)

    hppp_surface = np.full_like(intensity, lambda_hat)
    hppp_surface[np.isnan(intensity)] = np.nan

    if intensity_plot_scale == "relative":
        def log_relative(surface: np.ndarray) -> np.ndarray:
            mean_value = np.nanmean(surface)
            relative = surface / max(mean_value, MIN_LAMBDA)
            return np.log(np.clip(relative, MIN_LAMBDA, None))

        hppp_plot = log_relative(hppp_surface)
        intensity_plot = log_relative(intensity)
        nonlinear_intensity_plot = log_relative(nonlinear_intensity)
        colorbar_label = "log relative intensity (0 = panel mean)"
        cmap = "coolwarm"
        norm_values = np.concatenate(
            [
                intensity_plot[np.isfinite(intensity_plot)],
                nonlinear_intensity_plot[np.isfinite(nonlinear_intensity_plot)],
            ]
        )
        span = np.nanpercentile(np.abs(norm_values), 98)
        if not np.isfinite(span) or span <= 0:
            span = 0.1
        vmin, vmax = -span, span
        # Use an even number of contour breaks so zero is not itself a break;
        # otherwise a perfectly constant HPPP surface can render with artifacts.
        levels = np.linspace(vmin, vmax, 30)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    else:
        scale, colorbar_label = intensity_display_scale(analysis_crs)
        if experiment.use_temporal:
            colorbar_label = f"{colorbar_label} / day"
        hppp_plot = hppp_surface * scale
        intensity_plot = intensity * scale
        nonlinear_intensity_plot = nonlinear_intensity * scale
        cmap = "viridis"
        combined_values = np.concatenate(
            [
                hppp_plot[np.isfinite(hppp_plot)],
                intensity_plot[np.isfinite(intensity_plot)],
                nonlinear_intensity_plot[np.isfinite(nonlinear_intensity_plot)],
            ]
        )
        vmin = np.nanmin(combined_values)
        vmax = np.nanmax(combined_values)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1.0
        levels = np.linspace(vmin, vmax, 30)
        norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 6),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    hppp_value = float(np.nanmedian(hppp_plot))
    hppp_color = plt.get_cmap(cmap)(norm(hppp_value))
    plot_window.plot(
        ax=axes[0],
        color=hppp_color,
        edgecolor="none",
        alpha=0.85,
        zorder=1,
    )
    axes[0].scatter(plot_points.geometry.x, plot_points.geometry.y, s=6, color="black", alpha=0.25, zorder=5)
    day_label = f" (day {plot_day_of_year})" if experiment.use_temporal else ""
    axes[0].set_title(f"HPPP: Constant Intensity{day_label}")
    apply_map_axes(axes[0], plot_window, plot_crs)

    im = axes[1].contourf(
        gx_plot,
        gy_plot,
        intensity_plot,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
    )
    axes[1].scatter(plot_points.geometry.x, plot_points.geometry.y, s=6, color="black", alpha=0.25, zorder=5)
    axes[1].set_title(f"Linear IPPP: Fitted Intensity{day_label}")
    apply_map_axes(axes[1], plot_window, plot_crs)

    im = axes[2].contourf(
        gx_plot,
        gy_plot,
        nonlinear_intensity_plot,
        levels=levels,
        cmap=cmap,
        norm=norm,
        extend="both",
    )
    axes[2].scatter(plot_points.geometry.x, plot_points.geometry.y, s=6, color="black", alpha=0.25, zorder=5)
    axes[2].set_title(f"Nonlinear IPPP: Fitted Intensity{day_label}")
    apply_map_axes(axes[2], plot_window, plot_crs)

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
    output_name: str = "wood_thrush_linear_ippp_pearson_residuals.png",
    title: str = "Pearson Residual Surface: Linear IPPP",
) -> None:
    if experiment.quad_keep_mask is None:
        raise ValueError("Main quadrature mask is unavailable for residual plotting.")

    if experiment.quad_spatial_index_np is not None and len(pearson_resid) != experiment.quad_keep_mask.sum():
        spatial_resid = np.full(experiment.quad_keep_mask.sum(), np.nan, dtype=np.float64)
        for spatial_index in np.unique(experiment.quad_spatial_index_np):
            values = pearson_resid[experiment.quad_spatial_index_np == spatial_index]
            spatial_resid[int(spatial_index)] = np.nanmean(values)
    else:
        spatial_resid = pearson_resid

    resid_grid = np.full(
        experiment.grid_size * experiment.grid_size,
        np.nan,
        dtype=np.float64,
    )
    resid_grid[experiment.quad_keep_mask] = spatial_resid
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
    ax.set_title(title)
    apply_map_axes(ax, plot_window, plot_crs)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson residual", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(image_dir / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def average_spatial_intensity_by_day(
    experiment: WoodThrushExperiment,
    model: nn.Module,
    day_of_year: np.ndarray,
) -> np.ndarray:
    if experiment.quad_spatial_raw_np is None or experiment.quad_spatial_weights_np is None:
        raise ValueError("Main spatial quadrature grid is unavailable.")

    values = []
    spatial_weights = experiment.quad_spatial_weights_np
    for day in day_of_year:
        phase = phase_from_day_of_year(int(day))
        phases = np.full(len(experiment.quad_spatial_raw_np), phase, dtype=np.float64)
        covariates = (
            experiment.covariate_sampler.sample(experiment.quad_spatial_raw_np, experiment.coord_crs)
            if experiment.use_covariates
            else None
        )
        features = experiment.make_features(
            experiment.quad_spatial_raw_np,
            temporal_phase=phases,
            covariates_raw=covariates,
        )
        with torch.no_grad():
            lambda_vals = model(torch.tensor(features, dtype=torch.float32)).numpy().ravel()
        values.append(float(np.sum(spatial_weights * lambda_vals) / np.sum(spatial_weights)))
    return np.asarray(values)


def plot_temporal_intensity_curve(
    experiment: WoodThrushExperiment,
    lambda_hat: float,
    lin_model: LinearIPPP,
    nonlinear_model: NonlinearIPPP,
    analysis_crs,
    image_dir: Path,
) -> None:
    if not experiment.use_temporal:
        return

    days = np.arange(1, 366)
    scale, colorbar_label = intensity_display_scale(analysis_crs)
    colorbar_label = f"{colorbar_label} / day"
    hppp = np.full(len(days), lambda_hat * scale)
    linear = average_spatial_intensity_by_day(experiment, lin_model, days) * scale
    nonlinear = average_spatial_intensity_by_day(experiment, nonlinear_model, days) * scale

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(days, hppp, label="HPPP", linestyle="--", color="black")
    ax.plot(days, linear, label="Linear IPPP", color="steelblue")
    ax.plot(days, nonlinear, label="Nonlinear IPPP", color="darkorange")
    ax.set_title("Average Fitted Intensity by Day of Year")
    ax.set_xlabel("Day of year")
    ax.set_ylabel(colorbar_label)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_dir / "wood_thrush_temporal_intensity_curve.png", dpi=300, bbox_inches="tight")
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
    if "Nonlinear_heldout_loglik" in cv_df.columns:
        ax.plot(
            cv_df["fold"],
            cv_df["Nonlinear_heldout_loglik"],
            marker="o",
            label="Nonlinear IPPP",
        )
    ax.set_title("Spatial Block Cross-Validation")
    ax.set_xlabel("Held-out fold")
    ax.set_ylabel("Held-out IPPP log-likelihood")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_dir / "wood_thrush_spatial_block_cv.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_simulation_diagnostics(
    sim_df: pd.DataFrame,
    image_dir: Path,
    output_name: str = "wood_thrush_simulated_total_count.png",
    title: str = "Simulation Diagnostic: Total Count",
) -> None:
    observed_total = float(sim_df["observed_total"].iloc[0])
    expected_total = float(sim_df["expected_total"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(sim_df["total_count"], bins=30, color="steelblue", alpha=0.75)
    ax.axvline(observed_total, color="black", linewidth=2, label="Observed")
    ax.axvline(expected_total, color="darkorange", linestyle="--", linewidth=2, label="Expected")
    ax.set_title(title)
    ax.set_xlabel("Simulated total observations")
    ax.set_ylabel("Simulation frequency")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_dir / output_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_inhomogeneous_k(
    k_df: pd.DataFrame,
    image_dir: Path,
    output_name: str = "wood_thrush_inhomogeneous_k.png",
    title: str = "Approximate Inhomogeneous K-Function",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(k_df["radius"], k_df["K_inhom"], label="Estimated Kinhom")
    ax.plot(k_df["radius"], k_df["Poisson_theoretical"], linestyle="--", label="Poisson")
    ax.set_title(title)
    ax.set_xlabel("Radius")
    ax.set_ylabel("K(r)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(image_dir / output_name, dpi=300, bbox_inches="tight")
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
    covariate_names = selected_covariates(args)
    covariate_sampler = None
    if covariate_names:
        covariate_sampler = RasterCovariateSampler(
            Path(args.covariate_raster),
            selected_covariates=covariate_names,
            canopy_prefix=args.canopy_prefix,
        )
        print(f"Using spatial covariates: {', '.join(covariate_names)}")

    if args.use_temporal:
        temporal_phase_np, temporal_duration = temporal_phase_from_dates(points, args.date_column)
        print(
            f"Using cyclic day-of-year terms from {args.date_column}; "
            f"space-time exposure covers {temporal_duration:.0f} days."
        )
    else:
        temporal_phase_np = None
        temporal_duration = 1.0
    plot_temporal_phase = (
        phase_from_day_of_year(args.plot_day_of_year) if args.use_temporal else None
    )
    if covariate_sampler is not None and temporal_phase_np is not None:
        print("Using combined spatial covariate + cyclic temporal model features.")

    experiment = WoodThrushExperiment(
        coords_np=coords_np,
        window_geom=window_geom,
        grid_size=args.grid_size,
        coord_crs=points.crs,
        covariate_sampler=covariate_sampler,
        temporal_phase_np=temporal_phase_np,
        temporal_duration=temporal_duration,
        temporal_bins=args.temporal_bins,
    )
    analysis_crs = points.crs
    plot_crs = CRS.from_user_input(args.plot_crs)
    plot_window = gpd.GeoSeries([window_geom], crs=analysis_crs).to_crs(plot_crs)

    lambda_hat = experiment.n / experiment.measure
    loglik_hppp = experiment.n * np.log(lambda_hat) - lambda_hat * experiment.measure

    const_model = experiment.fit(
        ConstantIPPP(lambda_hat),
        epochs=args.epochs_constant,
        lr=args.lr,
    )
    loglik_const = experiment.bt_loglik(const_model)

    lin_model = experiment.fit(
        LinearIPPP(input_dim=experiment.input_dim, lambda_init=lambda_hat),
        epochs=args.epochs_linear,
        lr=args.lr,
    )
    loglik_lin = experiment.bt_loglik(lin_model)

    nonlinear_model = experiment.fit(
        NonlinearIPPP(
            input_dim=experiment.input_dim,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            lambda_init=lambda_hat,
            dropout=args.dropout,
        ),
        epochs=args.epochs_nonlinear,
        lr=args.nonlinear_lr,
        weight_decay=args.nonlinear_weight_decay,
    )
    loglik_nonlinear = experiment.bt_loglik(nonlinear_model)

    k_hppp = 1
    k_const = count_parameters(const_model)
    k_lin = count_parameters(lin_model)
    k_nonlinear = count_parameters(nonlinear_model)

    results_df = pd.DataFrame(
        {
            "Model": ["HPPP", "Constant NN", "Linear IPPP", "Nonlinear IPPP"],
            "k": [k_hppp, k_const, k_lin, k_nonlinear],
            "BT_LogLik": [loglik_hppp, loglik_const, loglik_lin, loglik_nonlinear],
            "AIC": [
                aic(loglik_hppp, k_hppp),
                aic(loglik_const, k_const),
                aic(loglik_lin, k_lin),
                aic(loglik_nonlinear, k_nonlinear),
            ],
            "BIC": [
                bic(loglik_hppp, k_hppp, experiment.n),
                bic(loglik_const, k_const, experiment.n),
                bic(loglik_lin, k_lin, experiment.n),
                bic(loglik_nonlinear, k_nonlinear, experiment.n),
            ],
        }
    )
    print(results_df)

    lr_lin_vs_hppp = lr_stat(loglik_lin, loglik_hppp)
    p_lin_vs_hppp = chi2.sf(lr_lin_vs_hppp, df=experiment.input_dim)
    lr_nonlinear_vs_linear = lr_stat(loglik_nonlinear, loglik_lin)
    df_nonlinear_vs_linear = k_nonlinear - k_lin
    p_nonlinear_vs_linear = chi2.sf(
        lr_nonlinear_vs_linear,
        df=df_nonlinear_vs_linear,
    )

    print("\nLikelihood Ratio Test:")
    print(f"Linear vs HPPP LR stat: {lr_lin_vs_hppp:.4f}")
    print(f"Linear vs HPPP p-value: {p_lin_vs_hppp:.6g}")
    print("\nNonlinear vs Linear Comparison:")
    print(f"Nonlinear vs Linear LR-like stat: {lr_nonlinear_vs_linear:.4f}")
    print(f"Nonlinear vs Linear df: {df_nonlinear_vs_linear}")
    print(f"Nonlinear vs Linear p-value: {p_nonlinear_vs_linear:.6g}")

    print("\nSanity checks:")
    print(f"Observations: {experiment.n:,}")
    print(f"Window area: {experiment.area:.4f}")
    if experiment.use_covariates:
        print(f"Covariates: {', '.join(experiment.covariate_names)}")
    if experiment.use_temporal:
        print(f"Temporal duration days: {experiment.temporal_duration:.0f}")
        print(f"Space-time measure: {experiment.measure:.4f}")
    print(f"lambda_hat: {lambda_hat:.6e}")
    print(f"HPPP loglik: {loglik_hppp:.4f}")
    print(f"Constant NN loglik: {loglik_const:.4f}")
    print(f"Linear IPPP loglik: {loglik_lin:.4f}")
    print(f"Nonlinear IPPP loglik: {loglik_nonlinear:.4f}")

    print("\nCoefficients:")
    weight = lin_model.linear.weight.detach().numpy()
    print(f"Linear weight: {weight}")
    print(f"Linear bias: {lin_model.linear.bias.item()}")
    formula_terms = [
        f"{weight[0, 0]:.4f}*x",
        f"{weight[0, 1]:.4f}*y",
    ]
    for cov_index, covariate_name in enumerate(experiment.covariate_names):
        formula_terms.append(f"{weight[0, 2 + cov_index]:.4f}*{covariate_name}")
    if experiment.use_temporal:
        temporal_start = 2 + len(experiment.covariate_names)
        formula_terms.extend(
            [
                f"{weight[0, temporal_start]:.4f}*sin_doy",
                f"{weight[0, temporal_start + 1]:.4f}*cos_doy",
            ]
        )
    print(
        "Fitted model is approximately: "
        f"{'lambda(s,t)' if experiment.use_temporal else 'lambda(s)'} = "
        f"exp({lin_model.linear.bias.item():.4f} + "
        + " + ".join(formula_terms)
        + ")"
    )

    raw_resid, pearson_resid, expected_counts, observed_counts = (
        experiment.berman_turner_residuals(lin_model)
    )
    (
        raw_resid_nonlinear,
        pearson_resid_nonlinear,
        expected_counts_nonlinear,
        observed_counts_nonlinear,
    ) = experiment.berman_turner_residuals(nonlinear_model)
    print("\nSpatial Residual Diagnostics:")
    print("Linear IPPP:")
    print(f"Mean raw residual: {raw_resid.mean():.6f}")
    print(f"Mean Pearson residual: {pearson_resid.mean():.6f}")
    print(f"Pearson residual SD: {pearson_resid.std():.6f}")
    print(f"Observed total count: {observed_counts.sum():.0f}")
    print(f"Expected total count: {expected_counts.sum():.4f}")
    print("Nonlinear IPPP:")
    print(f"Mean raw residual: {raw_resid_nonlinear.mean():.6f}")
    print(f"Mean Pearson residual: {pearson_resid_nonlinear.mean():.6f}")
    print(f"Pearson residual SD: {pearson_resid_nonlinear.std():.6f}")
    print(f"Observed total count: {observed_counts_nonlinear.sum():.0f}")
    print(f"Expected total count: {expected_counts_nonlinear.sum():.4f}")

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
        epochs_nonlinear=args.epochs_nonlinear,
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
        dropout=args.dropout,
        lr=args.lr,
        nonlinear_lr=args.nonlinear_lr,
        nonlinear_weight_decay=args.nonlinear_weight_decay,
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
                "Nonlinear_heldout_loglik",
            ]
        ].sum()
    )

    simulation_df = run_simulation_diagnostics(
        experiment=experiment,
        model=lin_model,
        n_simulations=args.simulation_count,
    )
    simulation_df["model"] = "Linear IPPP"
    simulation_nonlinear_df = run_simulation_diagnostics(
        experiment=experiment,
        model=nonlinear_model,
        n_simulations=args.simulation_count,
    )
    simulation_nonlinear_df["model"] = "Nonlinear IPPP"
    observed_total = float(simulation_df["observed_total"].iloc[0])
    simulated_totals = simulation_df["total_count"].to_numpy()
    print("\nSimulation Diagnostics:")
    print("Linear IPPP:")
    print(f"Observed total count: {observed_total:.0f}")
    print(f"Simulated total mean: {simulated_totals.mean():.2f}")
    print(f"Simulated total 2.5%-97.5%: {np.quantile(simulated_totals, [0.025, 0.975])}")
    print(f"Two-sided simulation p-value: {two_sided_simulation_pvalue(simulated_totals, observed_total):.4f}")
    observed_total_nonlinear = float(simulation_nonlinear_df["observed_total"].iloc[0])
    simulated_totals_nonlinear = simulation_nonlinear_df["total_count"].to_numpy()
    print("Nonlinear IPPP:")
    print(f"Observed total count: {observed_total_nonlinear:.0f}")
    print(f"Simulated total mean: {simulated_totals_nonlinear.mean():.2f}")
    print(
        "Simulated total 2.5%-97.5%: "
        f"{np.quantile(simulated_totals_nonlinear, [0.025, 0.975])}"
    )
    print(
        "Two-sided simulation p-value: "
        f"{two_sided_simulation_pvalue(simulated_totals_nonlinear, observed_total_nonlinear):.4f}"
    )

    k_df = inhomogeneous_k_diagnostic(
        experiment=experiment,
        model=lin_model,
        n_radii=args.k_radii,
    )
    k_df["model"] = "Linear IPPP"
    k_nonlinear_df = inhomogeneous_k_diagnostic(
        experiment=experiment,
        model=nonlinear_model,
        n_radii=args.k_radii,
    )
    k_nonlinear_df["model"] = "Nonlinear IPPP"
    print("\nInhomogeneous K Diagnostic:")
    print("Linear IPPP:")
    print(k_df.head())
    print("Nonlinear IPPP:")
    print(k_nonlinear_df.head())

    results_df.to_csv(image_dir / "wood_thrush_model_comparison.csv", index=False)
    grid_sensitivity_df.to_csv(
        image_dir / "wood_thrush_grid_sensitivity.csv",
        index=False,
    )
    cv_df.to_csv(image_dir / "wood_thrush_spatial_block_cv.csv", index=False)
    simulation_df.to_csv(image_dir / "wood_thrush_simulation_diagnostics.csv", index=False)
    simulation_nonlinear_df.to_csv(
        image_dir / "wood_thrush_nonlinear_simulation_diagnostics.csv",
        index=False,
    )
    k_df.to_csv(image_dir / "wood_thrush_inhomogeneous_k.csv", index=False)
    k_nonlinear_df.to_csv(
        image_dir / "wood_thrush_nonlinear_inhomogeneous_k.csv",
        index=False,
    )

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
        nonlinear_model,
        lambda_hat,
        args.plot_grid_size,
        plot_temporal_phase,
        args.plot_day_of_year,
        points,
        analysis_crs,
        plot_window,
        plot_crs,
        args.intensity_plot_scale,
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
    plot_residual_surface(
        experiment,
        pearson_resid_nonlinear,
        points,
        analysis_crs,
        plot_window,
        plot_crs,
        image_dir,
        output_name="wood_thrush_nonlinear_ippp_pearson_residuals.png",
        title="Pearson Residual Surface: Nonlinear IPPP",
    )
    plot_grid_sensitivity(grid_sensitivity_df, loglik_hppp, image_dir)
    plot_spatial_block_cv(cv_df, image_dir)
    plot_simulation_diagnostics(simulation_df, image_dir)
    plot_simulation_diagnostics(
        simulation_nonlinear_df,
        image_dir,
        output_name="wood_thrush_nonlinear_simulated_total_count.png",
        title="Simulation Diagnostic: Nonlinear IPPP Total Count",
    )
    plot_inhomogeneous_k(k_df, image_dir)
    plot_inhomogeneous_k(
        k_nonlinear_df,
        image_dir,
        output_name="wood_thrush_nonlinear_inhomogeneous_k.png",
        title="Approximate Inhomogeneous K-Function: Nonlinear IPPP",
    )
    plot_temporal_intensity_curve(
        experiment,
        lambda_hat,
        lin_model,
        nonlinear_model,
        analysis_crs,
        image_dir,
    )

    print(f"\nSaved figures to {image_dir}")


if __name__ == "__main__":
    main()

# TODO:
# - Adding temporal residual diagnostics beyond the annual-cycle curve
# - Reducing overfitting while keeping ecologically important components:
#   1. Add penalized linear IPPP fits. The current linear model is effectively
#      unpenalized, so add ridge/L2 or elastic-net penalties for covariate
#      coefficients. Temporal terms are strongly justified for a migratory
#      species and should be penalized less heavily, or separately, from static
#      spatial covariates.
#   2. Prefer structured nonlinear models over a generic fully flexible NN.
#      A better target is an additive log-intensity such as:
#
#          log lambda(s, t, z) =
#              intercept
#              + broad spatial trend
#              + cyclic seasonal effect
#              + covariate effects
#              + small nonlinear residual correction
#
#      This is closer to a GAM/IPPP than a black-box NN and should reduce the
#      model's ability to absorb clustered sampling artifacts as ecological
#      signal.
#   3. Implement a residual nonlinear model:
#
#          log lambda = linear_temporal_covariate_predictor + f_nn(features)
#
#      with a penalty that shrinks f_nn toward zero. This keeps the temporal
#      and covariate structure primary, and only lets the NN explain remaining
#      structure when it improves held-out spatial likelihood.
#   4. Use spatial-block validation for nonlinear early stopping. The current
#      nonlinear models improve in-sample BT likelihood but often lose held-out
#      block likelihood and total-count calibration. Stop training when
#      held-out spatial-block likelihood stops improving.
#   5. Add an optional total-count calibration penalty for experimental runs:
#
#          alpha * (expected_total - observed_total)^2 / observed_total
#
#      This is not pure IPPP maximum likelihood, but it directly targets the
#      observed failure mode where nonlinear models underpredict the total
#      expected count while fitting local structure.
#   6. Compare reduced feature sets to diagnose confounding:
#      temporal + coordinates only; temporal + covariates only; temporal +
#      covariates + broad spatial trend; temporal + covariates + penalized
#      nonlinear residual. This is especially important because canopy,
#      elevation, distance to coast, and x/y location can be spatially
#      confounded in North Carolina.
#   7. Add year-aligned canopy covariates. The current combined run used
#      `canopy_median` because that was the selected covariate name. To use
#      annual canopy aligned with each observation year, add a dynamic
#      covariate feature that selects the matching `tcc_<year>` band for
#      observation points and the matching year/bin during quadrature.
