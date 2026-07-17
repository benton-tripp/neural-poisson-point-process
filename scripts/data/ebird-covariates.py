"""Orchestrate versioned eBird environmental/access covariate builds.

Phase 1 implements a dry-run planner. It validates the source registry, AOI,
grid, temporal contract, and expected logical band inventory without fetching
source data.

Run from the project root:

    python scripts/data/ebird-covariates.py plan \
      --config config/ebird_covariates/nc_2020_2023_v1.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ebird_covariates.planner import (
    build_plan,
    load_json,
    print_plan_summary,
    write_plan,
)
from ebird_covariates.nlcd import (
    PRODUCTS as NLCD_PRODUCTS,
    parse_year_expression,
    resolve_catalog as resolve_nlcd_catalog,
    write_catalog as write_nlcd_catalog,
)
from ebird_covariates.raster_engine import (
    RESAMPLING_METHODS,
    load_band_inventory,
    load_plan,
    normalize_raster_to_tiles,
    safe_band_slug,
    write_band_inventory,
    write_logical_vrt,
)


DEFAULT_REGISTRY = Path(__file__).with_name("ebird_covariates") / "source_registry.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan and build the eBird CONUS covariate stack."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser(
        "plan",
        help="Validate a build config and write a tiled dry-run build plan.",
    )
    plan_parser.add_argument("--config", required=True, help="Build config JSON path.")
    plan_parser.add_argument(
        "--registry",
        default=str(DEFAULT_REGISTRY),
        help=f"Source registry JSON. Defaults to {DEFAULT_REGISTRY}.",
    )
    plan_parser.add_argument(
        "--output",
        help=(
            "Optional output plan path. Defaults to "
            "<output.root>/<build_id>/build_plan.json."
        ),
    )
    plan_parser.add_argument(
        "--no-write",
        action="store_true",
        help="Validate and print the summary without writing build_plan.json.",
    )

    normalize_parser = subparsers.add_parser(
        "normalize-raster",
        help="Warp one input raster band to fixed-grid COG tiles.",
    )
    normalize_parser.add_argument("--plan", required=True, help="build_plan.json path.")
    normalize_parser.add_argument("--input", required=True, help="Input raster path.")
    normalize_parser.add_argument("--band-id", required=True, help="Stable output band id.")
    normalize_parser.add_argument(
        "--source-band",
        type=int,
        default=1,
        help="One-based source raster band. Defaults to 1.",
    )
    normalize_parser.add_argument(
        "--resampling",
        choices=sorted(RESAMPLING_METHODS),
        default="bilinear",
        help="Reprojection resampling. Defaults to bilinear.",
    )
    normalize_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for source-band COG tiles and inventory JSON.",
    )
    normalize_parser.add_argument(
        "--inventory-output",
        help="Inventory JSON path. Defaults inside --output-dir.",
    )
    normalize_parser.add_argument(
        "--include-outside-aoi",
        action="store_true",
        help="Retain rectangular tile values outside the AOI boundary.",
    )
    normalize_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing COG tiles.",
    )

    assemble_parser = subparsers.add_parser(
        "assemble-vrt",
        help="Assemble one or more tiled band inventories into a logical VRT.",
    )
    assemble_parser.add_argument("--plan", required=True, help="build_plan.json path.")
    assemble_parser.add_argument(
        "--band-inventories",
        nargs="+",
        required=True,
        help="Band inventory JSON paths in desired output-band order.",
    )
    assemble_parser.add_argument(
        "--output",
        help="Output VRT. Defaults to the logical_raster path in the build plan.",
    )

    nlcd_parser = subparsers.add_parser(
        "catalog-nlcd",
        help="Resolve pinned Annual NLCD Collection 1.2 ScienceBase files.",
    )
    nlcd_parser.add_argument(
        "--plan",
        help="Optional build plan whose temporal start/end years should be used.",
    )
    nlcd_parser.add_argument(
        "--years",
        help="Years or ranges, for example 2020-2023 or 2020,2022. Overrides --plan.",
    )
    nlcd_parser.add_argument(
        "--products",
        nargs="+",
        choices=sorted(NLCD_PRODUCTS),
        help=(
            "Annual NLCD product codes to resolve. Defaults to the build-plan "
            "selection or the core land-cover and fractional-impervious products."
        ),
    )
    nlcd_parser.add_argument(
        "--output",
        default="data/ebird/covariates/raw/annual_nlcd/C1V2/catalog.json",
        help="Output catalog JSON path.",
    )
    nlcd_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="ScienceBase metadata request timeout in seconds.",
    )
    nlcd_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent metadata requests. Defaults to 4.",
    )
    return parser.parse_args()


def run_plan(args: argparse.Namespace) -> None:
    config_path = Path(args.config)
    registry_path = Path(args.registry)
    config = load_json(config_path)
    registry = load_json(registry_path)
    plan = build_plan(config, registry, config_path, registry_path)

    output_path: Path | None
    if args.no_write:
        output_path = None
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(plan["outputs"]["build_dir"]) / "build_plan.json"
    if output_path is not None:
        write_plan(plan, output_path)
    print_plan_summary(plan, output_path)


def run_normalize_raster(args: argparse.Namespace) -> None:
    plan = load_plan(Path(args.plan))
    output_dir = Path(args.output_dir)
    inventory = normalize_raster_to_tiles(
        plan=plan,
        source_path=Path(args.input),
        output_dir=output_dir,
        band_id=args.band_id,
        source_band=args.source_band,
        resampling=args.resampling,
        mask_outside_aoi=not args.include_outside_aoi,
        overwrite=args.overwrite,
    )
    inventory_path = (
        Path(args.inventory_output)
        if args.inventory_output
        else output_dir / f"{safe_band_slug(args.band_id)}__inventory.json"
    )
    write_band_inventory(inventory, inventory_path)
    print(
        f"Normalized {args.band_id}: {inventory['tile_count']} COG tiles, "
        f"{inventory['valid_cells']:,} valid cells"
    )
    print(f"Wrote band inventory to {inventory_path}")


def run_assemble_vrt(args: argparse.Namespace) -> None:
    plan = load_plan(Path(args.plan))
    inventories = [
        load_band_inventory(Path(path)) for path in args.band_inventories
    ]
    output_path = (
        Path(args.output) if args.output else Path(plan["outputs"]["logical_raster"])
    )
    write_logical_vrt(plan, inventories, output_path)
    print(f"Wrote {len(inventories)}-band logical raster to {output_path}")


def run_catalog_nlcd(args: argparse.Namespace) -> None:
    plan = load_plan(Path(args.plan)) if args.plan else None
    if args.years:
        years = parse_year_expression(args.years)
    elif plan:
        temporal = plan["temporal"]
        years = list(range(temporal["start_year"], temporal["end_year"] + 1))
    else:
        raise ValueError("catalog-nlcd requires --years or --plan.")

    nlcd_overrides = {}
    if plan:
        nlcd_sources = [
            source for source in plan["sources"] if source["id"] == "annual_nlcd"
        ]
        if len(nlcd_sources) != 1:
            raise ValueError("The build plan must contain exactly one annual_nlcd source.")
        nlcd_overrides = nlcd_sources[0].get("config_overrides", {})
    products = args.products or nlcd_overrides.get("products") or ["LndCov", "FctImp"]
    extra_years_by_product = {}
    if plan and not args.years:
        predecessor_years = nlcd_overrides.get("predecessor_land_cover_years", [])
        if predecessor_years:
            extra_years_by_product["LndCov"] = predecessor_years

    catalog = resolve_nlcd_catalog(
        years=years,
        products=products,
        extra_years_by_product=extra_years_by_product,
        timeout=args.timeout,
        max_workers=args.max_workers,
    )
    output_path = Path(args.output)
    write_nlcd_catalog(catalog, output_path)
    gib = catalog["total_source_bytes"] / (1024**3)
    print(
        f"Resolved Annual NLCD {catalog['release']}: {len(catalog['files'])} files, "
        f"{gib:.1f} GiB total"
    )
    for product in catalog["products"]:
        product_files = [
            file for file in catalog["files"] if file["product_code"] == product
        ]
        product_gib = sum(file["size_bytes"] for file in product_files) / (1024**3)
        print(f"  {product}: {len(product_files)} files, {product_gib:.1f} GiB")
    print(f"Acquisition status: {catalog['acquisition_status']}")
    print(f"Wrote catalog to {output_path}")


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        run_plan(args)
        return
    if args.command == "normalize-raster":
        run_normalize_raster(args)
        return
    if args.command == "assemble-vrt":
        run_assemble_vrt(args)
        return
    if args.command == "catalog-nlcd":
        run_catalog_nlcd(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
