"""
Create animated GIFs of fitted Wood Thrush temporal intensity surfaces.

Run from the project root, for example:

    python exp/wood_thrush_temporal_gifs.py --input data/wood_thrush_nc_2020_2023.geojson --boundary data/boundaries/nc_state_boundary.gpkg --analysis-crs EPSG:5070 --plot-crs EPSG:4326 --output-dir images/wood_thrush_nippp_temporal/gifs --models linear nonlinear
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from PIL import Image
from pyproj import CRS
import torch

from wood_thrush_nippp import (
    DEFAULT_INPUT,
    LinearIPPP,
    NonlinearIPPP,
    WoodThrushExperiment,
    build_window_geometry,
    filter_points_to_window,
    intensity_display_scale,
    load_points,
    phase_from_day_of_year,
    temporal_phase_from_dates,
    transform_xy,
    validate_bbox,
)


DEFAULT_OUTPUT_DIR = "images/wood_thrush_nippp_temporal/gifs"


def day_of_year_label(day: int) -> str:
    frame_date = date(2021, 1, 1) + timedelta(days=day - 1)
    return f"{frame_date.strftime('%B')} {frame_date.day}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render looping GIFs of fitted Wood Thrush temporal IPPP intensity surfaces."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Input Wood Thrush point GeoJSON. Defaults to {DEFAULT_INPUT}.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output GIFs. Defaults to {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("linear", "nonlinear"),
        default=["linear", "nonlinear"],
        help="Model GIFs to render. Defaults to linear nonlinear.",
    )
    parser.add_argument("--south", type=float, help="Optional WGS84 window south coordinate.")
    parser.add_argument("--north", type=float, help="Optional WGS84 window north coordinate.")
    parser.add_argument("--west", type=float, help="Optional WGS84 window west coordinate.")
    parser.add_argument("--east", type=float, help="Optional WGS84 window east coordinate.")
    parser.add_argument(
        "--boundary",
        help="Optional study-window boundary readable by GeoPandas.",
    )
    parser.add_argument(
        "--analysis-crs",
        default="EPSG:5070",
        help="CRS for fitting. Defaults to EPSG:5070.",
    )
    parser.add_argument(
        "--plot-crs",
        default="EPSG:4326",
        help="CRS used for GIF map frames. Defaults to EPSG:4326.",
    )
    parser.add_argument(
        "--date-column",
        default="queryDate",
        help="Date column used for cyclic day-of-year terms. Defaults to queryDate.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=100,
        help="Berman-Turner grid cells per dimension for fitting. Defaults to 100.",
    )
    parser.add_argument(
        "--temporal-bins",
        type=int,
        default=12,
        help="Number of annual-cycle bins in the space-time quadrature. Defaults to 12.",
    )
    parser.add_argument(
        "--plot-grid-size",
        type=int,
        default=120,
        help="Grid cells per dimension for GIF frames. Defaults to 120.",
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
        help="Training epochs for the nonlinear IPPP. Defaults to 10000.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate for the linear IPPP. Defaults to 1e-3.",
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
        "--hidden-dim",
        type=int,
        default=16,
        help="Hidden layer width for the nonlinear IPPP. Defaults to 16.",
    )
    parser.add_argument(
        "--hidden-layers",
        type=int,
        default=1,
        help="Number of hidden layers for the nonlinear IPPP. Defaults to 1.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
        help="Dropout rate for nonlinear hidden layers. Defaults to 0.10.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Animation frames per second. Defaults to 24.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Frame resolution in dots per inch. Defaults to 120.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=99.0,
        help="Upper percentile for fixed color scaling. Defaults to 99.",
    )
    parser.add_argument(
        "--point-window",
        type=int,
        default=0,
        help="Show observations within this many day-of-year days of each frame. Defaults to 0.",
    )
    parser.add_argument(
        "--show-boundary",
        action="store_true",
        help="Draw the study-window boundary line on each GIF frame.",
    )
    parser.add_argument(
        "--start-day",
        type=int,
        default=1,
        help="First day of year to render. Defaults to 1.",
    )
    parser.add_argument(
        "--end-day",
        type=int,
        default=365,
        help="Last day of year to render. Defaults to 365.",
    )
    return parser.parse_args()


def load_temporal_experiment(args: argparse.Namespace) -> tuple[
    WoodThrushExperiment,
    gpd.GeoDataFrame,
    gpd.GeoSeries,
    object,
    float,
]:
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

    temporal_phase_np, temporal_duration = temporal_phase_from_dates(points, args.date_column)
    coords_np = np.column_stack([points.geometry.x, points.geometry.y]).astype(np.float64)

    experiment = WoodThrushExperiment(
        coords_np=coords_np,
        window_geom=window_geom,
        grid_size=args.grid_size,
        coord_crs=points.crs,
        temporal_phase_np=temporal_phase_np,
        temporal_duration=temporal_duration,
        temporal_bins=args.temporal_bins,
    )
    plot_crs = CRS.from_user_input(args.plot_crs)
    plot_window = gpd.GeoSeries([window_geom], crs=points.crs).to_crs(plot_crs)
    lambda_hat = experiment.n / experiment.measure
    return experiment, points, plot_window, plot_crs, lambda_hat


def fit_models(
    args: argparse.Namespace,
    experiment: WoodThrushExperiment,
    lambda_hat: float,
) -> dict[str, torch.nn.Module]:
    models = {}
    if "linear" in args.models:
        print("Fitting temporal linear IPPP")
        models["linear"] = experiment.fit(
            LinearIPPP(input_dim=experiment.input_dim, lambda_init=lambda_hat),
            epochs=args.epochs_linear,
            lr=args.lr,
        )
    if "nonlinear" in args.models:
        print("Fitting temporal nonlinear IPPP")
        models["nonlinear"] = experiment.fit(
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
    return models


def day_distance(day_values: np.ndarray, day: int) -> np.ndarray:
    raw = np.abs(day_values - day)
    return np.minimum(raw, 365 - raw)


def compute_surfaces(
    experiment: WoodThrushExperiment,
    model: torch.nn.Module,
    plot_grid_size: int,
    scale: float,
    days: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    gx_plot = None
    gy_plot = None
    surfaces = []
    for day in days:
        phase = phase_from_day_of_year(day)
        gx, gy, surface = experiment.predict_intensity_surface(
            model,
            plot_grid_size,
            temporal_phase=phase,
        )
        if gx_plot is None or gy_plot is None:
            gx_plot, gy_plot = gx, gy
        surfaces.append(surface * scale)
    if gx_plot is None or gy_plot is None:
        raise ValueError("No intensity surfaces were generated.")
    return gx_plot, gy_plot, surfaces


def surfaces_color_scale(surfaces: list[np.ndarray], clip_percentile: float) -> tuple[float, float]:
    values = np.concatenate([surface[np.isfinite(surface)] for surface in surfaces])
    vmin = float(np.nanmin(values))
    vmax = float(np.nanpercentile(values, clip_percentile))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def figure_to_image(fig) -> Image.Image:
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
    return Image.fromarray(data[..., :3].copy())


def render_gif(
    model_name: str,
    experiment: WoodThrushExperiment,
    points: gpd.GeoDataFrame,
    plot_window: gpd.GeoSeries,
    plot_crs,
    gx: np.ndarray,
    gy: np.ndarray,
    surfaces: list[np.ndarray],
    days: np.ndarray,
    date_column: str,
    colorbar_label: str,
    output_path: Path,
    fps: float,
    dpi: int,
    clip_percentile: float,
    point_window: int,
    show_boundary: bool,
) -> None:
    source_crs = points.crs
    gx_plot, gy_plot = transform_xy(gx, gy, source_crs, plot_crs)
    plot_points = points.to_crs(plot_crs).copy()
    dates = pd.to_datetime(points[date_column], errors="coerce")
    day_values = dates.dt.dayofyear.to_numpy(dtype=np.int16)
    vmin, vmax = surfaces_color_scale(surfaces, clip_percentile)
    levels = np.linspace(vmin, vmax, 30)
    norm = Normalize(vmin=vmin, vmax=vmax)
    duration_ms = max(1, int(round(1000.0 / fps)))

    frames = []
    for frame_index, (day, surface) in enumerate(zip(days, surfaces), start=1):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=dpi)
        im = ax.contourf(
            gx_plot,
            gy_plot,
            surface,
            levels=levels,
            cmap="viridis",
            norm=norm,
            extend="max",
        )
        if show_boundary:
            plot_window.boundary.plot(ax=ax, color="black", linewidth=0.8, alpha=0.75)

        if point_window > 0:
            keep = day_distance(day_values, day) <= point_window
            if np.any(keep):
                ax.scatter(
                    plot_points.geometry.x[keep],
                    plot_points.geometry.y[keep],
                    s=8,
                    color="black",
                    alpha=0.35,
                    label=f"observations +/- {point_window} days",
                )
                ax.legend(loc="upper right", fontsize=7, frameon=True)

        bounds = plot_window.total_bounds
        x_pad = (bounds[2] - bounds[0]) * 0.03
        y_pad = (bounds[3] - bounds[1]) * 0.03
        ax.set_xlim(bounds[0] - x_pad, bounds[2] + x_pad)
        ax.set_ylim(bounds[1] - y_pad, bounds[3] + y_pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude" if CRS.from_user_input(plot_crs).to_epsg() == 4326 else "x")
        ax.set_ylabel("Latitude" if CRS.from_user_input(plot_crs).to_epsg() == 4326 else "y")
        ax.set_title(
            f"Wood Thrush {model_name.title()} Temporal IPPP - "
            f"{day_of_year_label(int(day))}"
        )
        ax.grid(alpha=0.2)
        cbar = fig.colorbar(im, ax=ax, shrink=0.82)
        cbar.set_label(colorbar_label, fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        fig.tight_layout()
        frames.append(figure_to_image(fig))
        plt.close(fig)

        if frame_index % 30 == 0 or frame_index == len(days):
            print(f"{model_name}: rendered frame {frame_index}/{len(days)}")

    if not frames:
        raise ValueError(f"No frames were rendered for {model_name}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    np.random.seed(19)
    torch.manual_seed(19)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.start_day < 1 or args.end_day > 365 or args.start_day > args.end_day:
        raise ValueError("--start-day and --end-day must define a valid range within 1..365.")
    days = np.arange(args.start_day, args.end_day + 1)

    experiment, points, plot_window, plot_crs, lambda_hat = load_temporal_experiment(args)
    models = fit_models(args, experiment, lambda_hat)
    scale, colorbar_label = intensity_display_scale(points.crs)
    colorbar_label = f"{colorbar_label} / day"

    for model_name, model in models.items():
        print(f"Computing daily surfaces for {model_name}")
        gx, gy, surfaces = compute_surfaces(
            experiment,
            model,
            args.plot_grid_size,
            scale,
            days,
        )
        render_gif(
            model_name=model_name,
            experiment=experiment,
            points=points,
            plot_window=plot_window,
            plot_crs=plot_crs,
            gx=gx,
            gy=gy,
            surfaces=surfaces,
            days=days,
            date_column=args.date_column,
            colorbar_label=colorbar_label,
            output_path=output_dir / f"wood_thrush_{model_name}_temporal_intensity.gif",
            fps=args.fps,
            dpi=args.dpi,
            clip_percentile=args.clip_percentile,
            point_window=args.point_window,
            show_boundary=args.show_boundary,
        )


if __name__ == "__main__":
    main()
