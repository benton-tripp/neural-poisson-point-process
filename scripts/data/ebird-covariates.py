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
import json
from pathlib import Path

from ebird_covariates.landfire import (
    SUPPORTED_PRODUCTS as LANDFIRE_PRODUCTS,
    resolve_catalog as resolve_landfire_catalog,
    write_catalog as write_landfire_catalog,
)
from ebird_covariates.landfire_attributes import (
    extract_attribute_tables as extract_landfire_attribute_tables,
    write_attribute_summary as write_landfire_attribute_summary,
)
from ebird_covariates.landfire_build import (
    PROFILE_DEFINITIONS as LANDFIRE_PROFILE_DEFINITIONS,
    build_landfire_state,
)
from ebird_covariates.landfire_crosswalk import (
    build_crosswalks as build_landfire_crosswalks,
)
from ebird_covariates.landfire_disturbance import build_disturbance_lookup
from ebird_covariates.landfire_disturbance_derive import (
    derive_landfire_disturbance,
)
from ebird_covariates.landfire_disturbance_qa import (
    plot_landfire_disturbance_preview,
    validate_landfire_disturbance_derivation,
    write_landfire_disturbance_validation,
)
from ebird_covariates.landfire_derive import derive_landfire
from ebird_covariates.landfire_export import export_landfire_tiles
from ebird_covariates.landfire_qa import (
    compare_landfire_releases,
    plot_landfire_tile_preview,
    validate_landfire_checklist_support,
    validate_landfire_derivation,
    write_landfire_checklist_support,
    write_landfire_derivation_validation,
    write_landfire_release_comparison,
)
from ebird_covariates.planner import (
    build_plan,
    load_json,
    print_plan_summary,
    write_plan,
)
from ebird_covariates.nlcd import (
    PRODUCTS as NLCD_PRODUCTS,
    parse_year_expression,
    register_aws_sources as register_nlcd_aws_sources,
    register_local_sources as register_nlcd_local_sources,
    resolve_catalog as resolve_nlcd_catalog,
    validate_registered_sources as validate_nlcd_registered_sources,
    write_catalog as write_nlcd_catalog,
    write_source_registration as write_nlcd_source_registration,
    write_source_validation as write_nlcd_source_validation,
)
from ebird_covariates.nlcd_derive import derive_nlcd
from ebird_covariates.nlcd_qa import (
    plot_nlcd_tile_preview,
    validate_nlcd_checklist_support,
    validate_nlcd_derivation,
    write_nlcd_checklist_support,
    write_nlcd_derivation_validation,
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

    nlcd_local_parser = subparsers.add_parser(
        "register-nlcd-local",
        help="Verify local Annual NLCD ZIP/TIFF inputs and write a source registration.",
    )
    nlcd_local_parser.add_argument("--catalog", required=True, help="NLCD catalog JSON.")
    nlcd_local_parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing catalogued ZIP archives or extracted TIFFs.",
    )
    nlcd_local_parser.add_argument(
        "--output",
        default="data/ebird/covariates/raw/annual_nlcd/C1V2/sources.local.json",
        help="Output source-registration JSON.",
    )
    nlcd_local_parser.add_argument(
        "--sha256",
        action="store_true",
        help="Calculate immutable SHA-256 hashes. This reads every source file.",
    )

    nlcd_aws_parser = subparsers.add_parser(
        "register-nlcd-aws",
        help="Write requester-pays AWS COG references from an NLCD catalog.",
    )
    nlcd_aws_parser.add_argument("--catalog", required=True, help="NLCD catalog JSON.")
    nlcd_aws_parser.add_argument(
        "--output",
        default="data/ebird/covariates/raw/annual_nlcd/C1V2/sources.aws.json",
        help="Output source-registration JSON.",
    )

    nlcd_validate_parser = subparsers.add_parser(
        "validate-nlcd-sources",
        help="Open registered NLCD rasters and validate their grid metadata.",
    )
    nlcd_validate_parser.add_argument(
        "--sources",
        required=True,
        help="Local or requester-pays AWS source-registration JSON.",
    )
    nlcd_validate_parser.add_argument(
        "--output",
        help="Validation JSON path. Defaults beside --sources.",
    )

    nlcd_derive_parser = subparsers.add_parser(
        "derive-nlcd",
        help="Derive tiled Annual NLCD ecological bands on the build-plan grid.",
    )
    nlcd_derive_parser.add_argument("--plan", required=True, help="build_plan.json path.")
    nlcd_derive_parser.add_argument(
        "--sources",
        required=True,
        help="Validated local or requester-pays AWS source-registration JSON.",
    )
    nlcd_derive_parser.add_argument(
        "--output-dir",
        help="Output directory. Defaults to <build_dir>/sources/annual_nlcd.",
    )
    nlcd_derive_parser.add_argument(
        "--years",
        help="Optional years/ranges overriding the plan, for example 2020-2023.",
    )
    nlcd_derive_parser.add_argument(
        "--neighborhoods-m",
        nargs="+",
        type=int,
        help="Optional circular-neighborhood radii overriding the plan.",
    )
    nlcd_derive_parser.add_argument(
        "--tile-ids",
        nargs="+",
        help=(
            "Optional plan tile IDs to derive. Use this for bounded pilot runs "
            "before processing the full AOI."
        ),
    )
    nlcd_derive_parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.8,
        help="Minimum valid source-area fraction for a derived value. Defaults to 0.8.",
    )
    nlcd_derive_parser.add_argument(
        "--no-vrt",
        action="store_true",
        help="Write COG tiles and inventories without assembling the source VRT.",
    )
    nlcd_derive_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing derived COG tiles.",
    )

    nlcd_derived_validate_parser = subparsers.add_parser(
        "validate-nlcd-derived",
        help="Validate derived Annual NLCD COGs and optionally render tile previews.",
    )
    nlcd_derived_validate_parser.add_argument(
        "--plan",
        required=True,
        help="build_plan.json path.",
    )
    nlcd_derived_validate_parser.add_argument(
        "--summary",
        required=True,
        help="annual_nlcd_summary.json path from derive-nlcd.",
    )
    nlcd_derived_validate_parser.add_argument(
        "--output",
        help=(
            "Validation JSON path. Defaults to "
            "<summary-dir>/diagnostics/annual_nlcd_validation.json."
        ),
    )
    nlcd_derived_validate_parser.add_argument(
        "--preview-tile-ids",
        nargs="+",
        help="Optional derived tile IDs for mapped QA previews.",
    )
    nlcd_derived_validate_parser.add_argument(
        "--preview-dir",
        help="Preview output directory. Defaults below the summary diagnostics dir.",
    )

    nlcd_checklist_parser = subparsers.add_parser(
        "validate-nlcd-checklist-support",
        help="Measure derived Annual NLCD support at processed checklist locations.",
    )
    nlcd_checklist_parser.add_argument(
        "--summary",
        required=True,
        help="annual_nlcd_summary.json path from derive-nlcd.",
    )
    nlcd_checklist_parser.add_argument(
        "--checklists",
        required=True,
        help="Processed checklist GeoParquet path.",
    )
    nlcd_checklist_parser.add_argument(
        "--output",
        help=(
            "Validation JSON path. Defaults to "
            "<summary-dir>/diagnostics/annual_nlcd_checklist_support.json."
        ),
    )
    nlcd_checklist_parser.add_argument(
        "--unsupported-output",
        help=(
            "Unsupported-checklist CSV path. Defaults to "
            "<summary-dir>/diagnostics/annual_nlcd_unsupported_checklists.csv."
        ),
    )

    landfire_parser = subparsers.add_parser(
        "catalog-landfire",
        help="Resolve release-pinned LANDFIRE layers and validate public ImageServers.",
    )
    landfire_parser.add_argument(
        "--plan",
        required=True,
        help="Build plan containing the LANDFIRE temporal/release policy.",
    )
    landfire_parser.add_argument(
        "--vegetation-releases",
        nargs="+",
        help="Optional LFYYYY releases overriding the build plan.",
    )
    landfire_parser.add_argument(
        "--products",
        nargs="+",
        choices=sorted(LANDFIRE_PRODUCTS),
        help="Optional product acronyms overriding the build plan.",
    )
    landfire_parser.add_argument(
        "--output",
        default="data/ebird/covariates/raw/landfire/catalog.json",
        help="Output catalog JSON path.",
    )
    landfire_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="LFPS metadata request timeout in seconds. Defaults to 60.",
    )
    landfire_parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Concurrent ImageServer metadata requests. Defaults to 6.",
    )

    landfire_attributes_parser = subparsers.add_parser(
        "catalog-landfire-attributes",
        help="Catalog release-specific class tables from official ImageServers.",
    )
    landfire_attributes_parser.add_argument(
        "--catalog",
        required=True,
        help="LANDFIRE catalog JSON from catalog-landfire.",
    )
    landfire_attributes_parser.add_argument(
        "--layers",
        nargs="+",
        help="Optional exact layer names, for example LF2023_EVT.",
    )
    landfire_attributes_parser.add_argument(
        "--output-dir",
        default="data/ebird/covariates/raw/landfire/attributes",
        help="Output directory for raw JSON, normalized CSV, and summary files.",
    )
    landfire_attributes_parser.add_argument(
        "--summary-output",
        help="Summary JSON path. Defaults below --output-dir.",
    )
    landfire_attributes_parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP request timeout in seconds. Defaults to 120.",
    )
    landfire_attributes_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace previously cataloged attribute-table files.",
    )

    landfire_crosswalk_parser = subparsers.add_parser(
        "build-landfire-crosswalks",
        help="Build release-aware model lookups from LANDFIRE class tables.",
    )
    landfire_crosswalk_parser.add_argument(
        "--attributes-summary",
        required=True,
        help="landfire_attribute_tables.json from catalog-landfire-attributes.",
    )
    landfire_crosswalk_parser.add_argument(
        "--output-dir",
        default="data/ebird/covariates/raw/landfire/crosswalks",
        help="Output directory for crosswalk CSVs and their summary.",
    )
    landfire_crosswalk_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace previously generated crosswalk files.",
    )

    disturbance_lookup_parser = subparsers.add_parser(
        "build-landfire-disturbance-lookup",
        help="Build annual event/support lookups from official Dist tables.",
    )
    disturbance_lookup_parser.add_argument(
        "--attributes-summary",
        required=True,
        help="Disturbance landfire_attribute_tables.json path.",
    )
    disturbance_lookup_parser.add_argument(
        "--output-dir",
        default="data/ebird/covariates/raw/landfire/disturbance_lookup",
        help="Output directory for the disturbance lookup and summary.",
    )
    disturbance_lookup_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace previously generated disturbance lookup files.",
    )

    landfire_export_parser = subparsers.add_parser(
        "export-landfire",
        help="Export and validate bounded raw LANDFIRE ImageServer tiles.",
    )
    landfire_export_parser.add_argument(
        "--plan",
        required=True,
        help="build_plan.json path.",
    )
    landfire_export_parser.add_argument(
        "--catalog",
        required=True,
        help="LANDFIRE catalog JSON from catalog-landfire.",
    )
    landfire_export_parser.add_argument(
        "--crosswalk-summary",
        required=True,
        help="landfire_crosswalk_summary.json path.",
    )
    landfire_export_parser.add_argument(
        "--tile-ids",
        nargs="+",
        required=True,
        help="One or more exact build-plan tile IDs.",
    )
    landfire_export_parser.add_argument(
        "--layers",
        nargs="+",
        required=True,
        help="One or more exact LANDFIRE layer names.",
    )
    landfire_export_parser.add_argument(
        "--buffer-m",
        type=float,
        default=5000.0,
        help="Source buffer around each plan tile in meters. Defaults to 5000.",
    )
    landfire_export_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for raw TIFFs and export summary.",
    )
    landfire_export_parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-export HTTP timeout in seconds. Defaults to 300.",
    )
    landfire_export_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace previously exported raw TIFFs.",
    )

    landfire_derive_parser = subparsers.add_parser(
        "derive-landfire",
        help="Derive model-scale LANDFIRE bands from one validated export group.",
    )
    landfire_derive_parser.add_argument(
        "--plan",
        required=True,
        help="build_plan.json path.",
    )
    landfire_derive_parser.add_argument(
        "--export-summary",
        required=True,
        help="landfire_export_summary.json for one tile and release.",
    )
    landfire_derive_parser.add_argument(
        "--crosswalk-summary",
        required=True,
        help="landfire_crosswalk_summary.json path.",
    )
    landfire_derive_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for derived COGs, inventories, VRT, and summary.",
    )
    landfire_derive_parser.add_argument(
        "--neighborhoods-m",
        type=int,
        nargs="+",
        help="Neighborhood radii in meters. Defaults to the build plan.",
    )
    landfire_derive_parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.8,
        help="Minimum valid source-area fraction. Defaults to 0.8.",
    )
    landfire_derive_parser.add_argument(
        "--minimum-lifeform-fraction",
        type=float,
        default=0.01,
        help="Minimum named-lifeform fraction for conditional means. Defaults to 0.01.",
    )
    landfire_derive_parser.add_argument(
        "--no-vrt",
        action="store_true",
        help="Do not assemble the derived logical VRT.",
    )
    landfire_derive_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace previously derived LANDFIRE COGs.",
    )
    landfire_validate_parser = subparsers.add_parser(
        "validate-landfire-derived",
        help="Validate derived LANDFIRE COGs, VRT, ranges, and class closure.",
    )
    landfire_validate_parser.add_argument(
        "--plan",
        required=True,
        help="build_plan.json path.",
    )
    landfire_validate_parser.add_argument(
        "--summary",
        required=True,
        help="landfire_derived_summary.json path from derive-landfire.",
    )
    landfire_validate_parser.add_argument(
        "--output",
        help="Validation JSON path. Defaults below the summary diagnostics dir.",
    )
    landfire_validate_parser.add_argument(
        "--preview",
        action="store_true",
        help="Render a mapped four-panel QA preview for the derived tile.",
    )
    landfire_validate_parser.add_argument(
        "--preview-output",
        help="Preview PNG path. Defaults below the summary diagnostics dir.",
    )

    landfire_checklist_parser = subparsers.add_parser(
        "validate-landfire-checklist-support",
        help="Measure one completed LANDFIRE release at checklist locations.",
    )
    landfire_checklist_parser.add_argument(
        "--plan",
        required=True,
        help="build_plan.json path.",
    )
    landfire_checklist_parser.add_argument(
        "--manifest",
        required=True,
        help="landfire_state_build_manifest.json path.",
    )
    landfire_checklist_parser.add_argument(
        "--checklists",
        required=True,
        help="Processed checklist GeoParquet path.",
    )
    landfire_checklist_parser.add_argument(
        "--release",
        required=True,
        help="Completed vegetation release to inspect, for example LF2016.",
    )
    landfire_checklist_parser.add_argument(
        "--output",
        help="Validation JSON path. Defaults below the state manifest diagnostics dir.",
    )
    landfire_checklist_parser.add_argument(
        "--unsupported-output",
        help="Unsupported-checklist CSV path. Defaults beside the validation JSON.",
    )

    landfire_release_compare_parser = subparsers.add_parser(
        "compare-landfire-releases",
        help="Compare two completed LANDFIRE vegetation releases.",
    )
    landfire_release_compare_parser.add_argument(
        "--plan",
        required=True,
        help="build_plan.json path.",
    )
    landfire_release_compare_parser.add_argument(
        "--manifest",
        required=True,
        help="landfire_state_build_manifest.json path.",
    )
    landfire_release_compare_parser.add_argument(
        "--baseline-release",
        required=True,
        help="Earlier completed vegetation release, for example LF2016.",
    )
    landfire_release_compare_parser.add_argument(
        "--comparison-release",
        required=True,
        help="Later completed vegetation release, for example LF2022.",
    )
    landfire_release_compare_parser.add_argument(
        "--tile-ids",
        nargs="+",
        help="Tiles for pixel comparisons. Defaults to all plan tiles.",
    )
    landfire_release_compare_parser.add_argument(
        "--output",
        help="Comparison JSON path. Defaults below the state manifest diagnostics dir.",
    )
    landfire_release_compare_parser.add_argument(
        "--metrics-output",
        help="Per-band comparison CSV path. Defaults beside the comparison JSON.",
    )

    disturbance_derive_parser = subparsers.add_parser(
        "derive-landfire-disturbance",
        help="Derive annual LANDFIRE disturbance fractions for one plan tile.",
    )
    disturbance_derive_parser.add_argument(
        "--plan",
        required=True,
        help="build_plan.json path.",
    )
    disturbance_derive_parser.add_argument(
        "--export-summary",
        required=True,
        help="Dist20-Dist23 landfire_export_summary.json for one tile.",
    )
    disturbance_derive_parser.add_argument(
        "--lookup-summary",
        required=True,
        help="landfire_disturbance_lookup_summary.json path.",
    )
    disturbance_derive_parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory for disturbance COGs, inventories, VRT, and summary.",
    )
    disturbance_derive_parser.add_argument(
        "--neighborhoods-m",
        type=int,
        nargs="+",
        help="Neighborhood radii in meters. Defaults to the build plan.",
    )
    disturbance_derive_parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.8,
        help="Minimum mappable terrestrial support fraction. Defaults to 0.8.",
    )
    disturbance_derive_parser.add_argument(
        "--no-vrt",
        action="store_true",
        help="Do not assemble the disturbance logical VRT.",
    )
    disturbance_derive_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace previously derived disturbance COGs.",
    )

    disturbance_validate_parser = subparsers.add_parser(
        "validate-landfire-disturbance-derived",
        help="Validate annual disturbance COGs, VRT, ranges, and support.",
    )
    disturbance_validate_parser.add_argument(
        "--plan",
        required=True,
        help="build_plan.json path.",
    )
    disturbance_validate_parser.add_argument(
        "--summary",
        required=True,
        help="landfire_disturbance_derived_summary.json path.",
    )
    disturbance_validate_parser.add_argument(
        "--output",
        help="Validation JSON path. Defaults below the summary diagnostics dir.",
    )
    disturbance_validate_parser.add_argument(
        "--preview",
        action="store_true",
        help="Render a four-year mapped disturbance preview.",
    )
    disturbance_validate_parser.add_argument(
        "--preview-output",
        help="Preview PNG path. Defaults below the summary diagnostics dir.",
    )

    state_build_parser = subparsers.add_parser(
        "build-landfire-state",
        help=(
            "Resume validated LANDFIRE tile/release units and assemble "
            "shared statewide profiles."
        ),
    )
    state_build_parser.add_argument(
        "--plan",
        required=True,
        help="build_plan.json path.",
    )
    state_build_parser.add_argument(
        "--catalog",
        required=True,
        help="LANDFIRE catalog JSON from catalog-landfire.",
    )
    state_build_parser.add_argument(
        "--vegetation-crosswalk-summary",
        required=True,
        help="landfire_crosswalk_summary.json path.",
    )
    state_build_parser.add_argument(
        "--disturbance-lookup-summary",
        required=True,
        help="landfire_disturbance_lookup_summary.json path.",
    )
    state_build_parser.add_argument(
        "--output-dir",
        required=True,
        help="State-build work, manifest, shared inventories, and profile VRTs.",
    )
    state_build_parser.add_argument(
        "--profiles",
        nargs="+",
        choices=list(LANDFIRE_PROFILE_DEFINITIONS),
        default=list(LANDFIRE_PROFILE_DEFINITIONS),
        help="Logical materialization profiles. Defaults to all profiles.",
    )
    state_build_parser.add_argument(
        "--tile-ids",
        nargs="+",
        help="Optional bounded plan-tile subset; defaults to every AOI tile.",
    )
    state_build_parser.add_argument(
        "--releases",
        nargs="+",
        help="Optional LFYYYY subset; defaults to every catalog release.",
    )
    state_build_parser.add_argument(
        "--components",
        nargs="+",
        choices=["vegetation", "disturbance"],
        help="Optional diagnostic component subset; defaults to both.",
    )
    state_build_parser.add_argument(
        "--buffer-m",
        type=float,
        default=5000.0,
        help="Raw source buffer around each plan tile. Defaults to 5000.",
    )
    state_build_parser.add_argument(
        "--minimum-coverage",
        type=float,
        default=0.8,
        help="Minimum valid/mappable source-area fraction. Defaults to 0.8.",
    )
    state_build_parser.add_argument(
        "--minimum-lifeform-fraction",
        type=float,
        default=0.01,
        help="Minimum lifeform support for conditional structure. Defaults to 0.01.",
    )
    state_build_parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-export HTTP timeout in seconds. Defaults to 300.",
    )
    state_build_parser.add_argument(
        "--max-units",
        type=int,
        help="Process at most this many incomplete work units this invocation.",
    )
    state_build_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write/validate the deterministic manifest without raster work.",
    )
    state_build_parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed units and continue processing independent units.",
    )
    state_build_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Intentionally replace otherwise reusable raw and derived units.",
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


def load_json_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON input does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_register_nlcd_local(args: argparse.Namespace) -> None:
    catalog = load_json_file(Path(args.catalog))
    registration = register_nlcd_local_sources(
        catalog=catalog,
        input_dir=Path(args.input_dir),
        calculate_sha256=args.sha256,
    )
    output_path = Path(args.output)
    write_nlcd_source_registration(registration, output_path)
    print(
        f"Registered {len(registration['sources'])} local Annual NLCD rasters "
        f"from {registration['input_dir']}"
    )
    print(f"Wrote source registration to {output_path}")


def run_register_nlcd_aws(args: argparse.Namespace) -> None:
    catalog = load_json_file(Path(args.catalog))
    registration = register_nlcd_aws_sources(catalog)
    output_path = Path(args.output)
    write_nlcd_source_registration(registration, output_path)
    print(
        f"Registered {len(registration['sources'])} requester-pays Annual NLCD "
        f"COG references in {registration['aws_region']}"
    )
    print("References are not opened until credentials are supplied to a build command.")
    print(f"Wrote source registration to {output_path}")


def run_validate_nlcd_sources(args: argparse.Namespace) -> None:
    source_path = Path(args.sources)
    registration = load_json_file(source_path)
    validation = validate_nlcd_registered_sources(registration)
    output_path = (
        Path(args.output)
        if args.output
        else source_path.with_name(source_path.stem + ".validation.json")
    )
    write_nlcd_source_validation(validation, output_path)
    print(
        f"Validated {validation['source_count']} Annual NLCD rasters; "
        f"land-cover grids aligned={validation['land_cover_grids_aligned']}"
    )
    print(f"Wrote source validation to {output_path}")


def run_derive_nlcd(args: argparse.Namespace) -> None:
    plan = load_plan(Path(args.plan))
    registration = load_json_file(Path(args.sources))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(plan["outputs"]["build_dir"]) / "sources" / "annual_nlcd"
    )
    summary = derive_nlcd(
        plan=plan,
        registration=registration,
        output_dir=output_dir,
        years=parse_year_expression(args.years) if args.years else None,
        neighborhoods_m=args.neighborhoods_m,
        tile_ids=args.tile_ids,
        minimum_coverage=args.minimum_coverage,
        write_vrt=not args.no_vrt,
        overwrite=args.overwrite,
        progress=True,
    )
    print(
        f"Derived Annual NLCD {summary['release']}: "
        f"{summary['band_count']} bands across {summary['tile_count']} plan tiles"
    )
    print(
        f"COG payload: {summary['derived_cog_count']:,} files, "
        f"{summary['derived_cog_mib']:.2f} MiB"
    )
    print(f"Elapsed time: {summary['elapsed_seconds']:.1f} seconds")
    if summary["logical_vrt"]:
        print(f"Wrote logical raster to {summary['logical_vrt']}")
    print(f"Wrote derivation summary to {output_dir / 'annual_nlcd_summary.json'}")


def run_validate_nlcd_derived(args: argparse.Namespace) -> None:
    plan = load_plan(Path(args.plan))
    summary_path = Path(args.summary)
    summary = load_json_file(summary_path)
    validation = validate_nlcd_derivation(plan, summary)
    diagnostics_dir = summary_path.parent / "diagnostics"
    output_path = (
        Path(args.output)
        if args.output
        else diagnostics_dir / "annual_nlcd_validation.json"
    )
    write_nlcd_derivation_validation(validation, output_path)
    maximum_error = validation["maximum_class_fraction_sum_error"]
    maximum_error_text = (
        f"{maximum_error:.3g}" if maximum_error is not None else "not calculated"
    )
    print(
        f"Validated derived Annual NLCD: {validation['band_count']} bands, "
        f"{validation['derived_cog_count']} COGs; "
        f"maximum class-fraction sum error={maximum_error_text}"
    )
    print(f"All checks passed: {validation['all_checks_passed']}")
    support_by_tile: dict[str, dict[int, float]] = {}
    for check in validation["class_fraction_sum_checks"]:
        tile_support = support_by_tile.setdefault(check["tile_id"], {})
        radius = int(check["radius_m"])
        tile_support[radius] = min(
            tile_support.get(radius, 1.0),
            float(check["supported_aoi_fraction"]),
        )
    for tile_id, support_by_radius in support_by_tile.items():
        formatted = ", ".join(
            f"r{radius}={fraction:.2%}"
            for radius, fraction in sorted(support_by_radius.items())
        )
        print(f"Minimum AOI support {tile_id}: {formatted}")
    print(f"Wrote derivation validation to {output_path}")

    if args.preview_tile_ids:
        preview_dir = (
            Path(args.preview_dir)
            if args.preview_dir
            else diagnostics_dir / "previews"
        )
        for tile_id in args.preview_tile_ids:
            preview_path = preview_dir / f"{tile_id}_annual_nlcd_preview.png"
            plot_nlcd_tile_preview(plan, summary, tile_id, preview_path)
            print(f"Wrote Annual NLCD preview to {preview_path}")

    if not validation["all_checks_passed"]:
        issue_summary = "; ".join(validation["issues"][:5])
        raise RuntimeError(
            f"Derived Annual NLCD validation failed with "
            f"{len(validation['issues'])} issue(s): {issue_summary}"
        )


def run_validate_nlcd_checklist_support(args: argparse.Namespace) -> None:
    summary_path = Path(args.summary)
    summary = load_json_file(summary_path)
    diagnostics_dir = summary_path.parent / "diagnostics"
    output_path = (
        Path(args.output)
        if args.output
        else diagnostics_dir / "annual_nlcd_checklist_support.json"
    )
    unsupported_output_path = (
        Path(args.unsupported_output)
        if args.unsupported_output
        else diagnostics_dir / "annual_nlcd_unsupported_checklists.csv"
    )
    validation, unsupported = validate_nlcd_checklist_support(
        summary,
        Path(args.checklists),
    )
    write_nlcd_checklist_support(
        validation,
        unsupported,
        output_path,
        unsupported_output_path,
    )
    print(
        f"Annual NLCD checklist support: "
        f"{validation['eligible_checklist_count']:,}/"
        f"{validation['checklist_count']:,} checklists eligible"
    )
    for record in validation["support_by_radius"]:
        fraction = record["supported_fraction"]
        fraction_text = f"{fraction:.4%}" if fraction is not None else "n/a"
        print(
            f"  r{record['radius_m']}: "
            f"{record['supported_checklists']:,}/"
            f"{record['eligible_checklists']:,} supported ({fraction_text}); "
            f"{record['unsupported_checklists']:,} unsupported"
        )
    print(f"Wrote checklist-support validation to {output_path}")
    print(f"Wrote unsupported checklist details to {unsupported_output_path}")


def run_catalog_landfire(args: argparse.Namespace) -> None:
    plan = load_plan(Path(args.plan))
    landfire_sources = [
        source for source in plan["sources"] if source["id"] == "landfire"
    ]
    if len(landfire_sources) != 1:
        raise ValueError("The build plan must contain exactly one landfire source.")
    overrides = landfire_sources[0].get("config_overrides", {})
    years = list(range(plan["temporal"]["start_year"], plan["temporal"]["end_year"] + 1))
    releases = args.vegetation_releases or overrides.get("releases")
    products = args.products or overrides.get("products")
    release_by_year = overrides.get("release_by_year")
    if args.vegetation_releases:
        release_by_year = None
    disturbance_years = overrides.get("disturbance_years", years)

    catalog = resolve_landfire_catalog(
        observation_years=years,
        vegetation_releases=releases,
        release_by_year=release_by_year,
        disturbance_years=disturbance_years,
        products=products,
        timeout=args.timeout,
        max_workers=args.max_workers,
    )
    output_path = Path(args.output)
    write_landfire_catalog(catalog, output_path)
    vegetation_count = sum(
        layer["role"] == "vegetation_release" for layer in catalog["layers"]
    )
    disturbance_count = sum(
        layer["role"] == "annual_disturbance" for layer in catalog["layers"]
    )
    mapping = ", ".join(
        f"{year}->{release}"
        for year, release in catalog["vegetation_release_by_year"].items()
    )
    print(
        f"Resolved LANDFIRE: {catalog['layer_count']} official layers "
        f"({vegetation_count} vegetation, {disturbance_count} disturbance)"
    )
    print(f"Vegetation release mapping: {mapping}")
    print(
        f"Validated public ImageServers: EPSG:{catalog['layers'][0]['wkid']} at "
        f"{catalog['layers'][0]['pixel_size_m'][0]:g} m; "
        f"all passed={catalog['all_services_validated']}"
    )
    print(f"Acquisition status: {catalog['acquisition_status']}")
    print(f"Wrote catalog to {output_path}")


def run_catalog_landfire_attributes(args: argparse.Namespace) -> None:
    catalog = load_json_file(Path(args.catalog))
    output_dir = Path(args.output_dir)
    summary = extract_landfire_attribute_tables(
        catalog=catalog,
        output_dir=output_dir,
        layer_names=args.layers,
        timeout=args.timeout,
        overwrite=args.overwrite,
    )
    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else output_dir / "landfire_attribute_tables.json"
    )
    write_landfire_attribute_summary(summary, summary_path)
    for record in summary["tables"]:
        print(
            f"  {record['layer_name']}: {record['row_count']:,} rows, "
            f"{record['field_count']} fields, "
            f"{record['raw_json_bytes']:,} response bytes"
        )
    print(
        f"Cataloged {summary['table_count']} LANDFIRE attribute tables "
        f"({summary['total_rows']:,} rows total)"
    )
    print(f"Wrote attribute-table summary to {summary_path}")


def run_build_landfire_crosswalks(args: argparse.Namespace) -> None:
    attribute_summary = load_json_file(Path(args.attributes_summary))
    summary = build_landfire_crosswalks(
        attribute_summary,
        Path(args.output_dir),
        overwrite=args.overwrite,
    )
    print(
        f"Built {summary['artifact_count']} LANDFIRE crosswalk artifacts "
        f"for {len(summary['model_classes'])} portable EVT classes"
    )
    print(
        "EVT release rows: "
        + ", ".join(
            f"{release}={rows:,}"
            for release, rows in summary["evt_release_rows"].items()
        )
    )
    print(
        f"Numeric structural classes: EVC={summary['evc_numeric_rows']:,}, "
        f"EVH={summary['evh_numeric_rows']:,}"
    )
    print(f"Wrote crosswalk summary to {summary['summary_path']}")


def run_build_landfire_disturbance_lookup(args: argparse.Namespace) -> None:
    summary = build_disturbance_lookup(
        load_json_file(Path(args.attributes_summary)),
        Path(args.output_dir),
        overwrite=args.overwrite,
    )
    for record in summary["year_records"]:
        print(
            f"  {record['year']} {record['layer_name']}: "
            f"{record['event_code_count']} event codes, "
            f"{record['water_mask_code_count']} Water mask codes, "
            f"{record['fill_code_count']} fill codes"
        )
    print(
        f"Built LANDFIRE disturbance lookup for "
        f"{len(summary['disturbance_years'])} years"
    )
    print(f"Wrote disturbance lookup summary to {summary['summary_path']}")


def run_export_landfire(args: argparse.Namespace) -> None:
    summary = export_landfire_tiles(
        plan=load_plan(Path(args.plan)),
        catalog=load_json_file(Path(args.catalog)),
        crosswalk_summary=load_json_file(Path(args.crosswalk_summary)),
        output_dir=Path(args.output_dir),
        tile_ids=args.tile_ids,
        layer_names=args.layers,
        buffer_m=args.buffer_m,
        timeout=args.timeout,
        overwrite=args.overwrite,
        progress=True,
    )
    for record in summary["exports"]:
        print(
            f"  {record['tile_id']} {record['layer_name']}: "
            f"{record['width']}x{record['height']}, "
            f"{record['unique_value_count']} values, "
            f"{record['coverage_fraction']:.2%} coverage, "
            f"{record['bytes'] / (1024**2):.1f} MiB"
        )
    print(
        f"Exported {summary['export_count']} validated LANDFIRE rasters; "
        f"all checks passed={summary['all_checks_passed']}"
    )
    print(f"Wrote export summary to {summary['summary_path']}")


def run_derive_landfire(args: argparse.Namespace) -> None:
    summary = derive_landfire(
        plan=load_plan(Path(args.plan)),
        export_summary=load_json_file(Path(args.export_summary)),
        crosswalk_summary=load_json_file(Path(args.crosswalk_summary)),
        output_dir=Path(args.output_dir),
        neighborhoods_m=args.neighborhoods_m,
        minimum_coverage=args.minimum_coverage,
        minimum_lifeform_fraction=args.minimum_lifeform_fraction,
        overwrite=args.overwrite,
        write_vrt=not args.no_vrt,
        progress=True,
    )
    print(
        f"Derived LANDFIRE {summary['release']} {summary['tile_id']}: "
        f"{summary['band_count']} bands, "
        f"{summary['derived_cog_count']} COGs, "
        f"{summary['derived_cog_bytes'] / (1024**2):.2f} MiB"
    )
    print(f"Wrote derivation summary to {summary['summary_path']}")


def run_validate_landfire_derived(args: argparse.Namespace) -> None:
    plan = load_plan(Path(args.plan))
    summary_path = Path(args.summary)
    summary = load_json_file(summary_path)
    validation = validate_landfire_derivation(plan, summary)
    diagnostics_dir = summary_path.parent / "diagnostics"
    output_path = (
        Path(args.output)
        if args.output
        else diagnostics_dir / "landfire_validation.json"
    )
    write_landfire_derivation_validation(validation, output_path)
    maximum_error = validation["maximum_evt_fraction_sum_error"]
    maximum_error_text = (
        f"{maximum_error:.3g}" if maximum_error is not None else "not calculated"
    )
    print(
        f"Validated derived LANDFIRE {validation['release']} "
        f"{validation['tile_id']}: {validation['band_count']} bands, "
        f"{validation['derived_cog_count']} COGs, "
        f"{validation['empty_band_count']} empty logical bands; "
        f"maximum EVT fraction-sum error={maximum_error_text}"
    )
    for check in validation["evt_fraction_sum_checks"]:
        support = check["supported_aoi_fraction"]
        support_text = f"{support:.2%}" if support is not None else "n/a"
        print(f"  r{check['radius_m']}: AOI support={support_text}")
    print(f"All checks passed: {validation['all_checks_passed']}")
    print(f"Wrote derivation validation to {output_path}")
    if args.preview:
        preview_path = (
            Path(args.preview_output)
            if args.preview_output
            else diagnostics_dir
            / f"{validation['tile_id']}_landfire_preview.png"
        )
        plot_landfire_tile_preview(plan, summary, preview_path)
        print(f"Wrote LANDFIRE preview to {preview_path}")
    if not validation["all_checks_passed"]:
        issue_summary = "; ".join(validation["issues"][:5])
        raise RuntimeError(
            f"Derived LANDFIRE validation failed with "
            f"{len(validation['issues'])} issue(s): {issue_summary}"
        )


def run_validate_landfire_checklist_support(args: argparse.Namespace) -> None:
    plan = load_plan(Path(args.plan))
    manifest_path = Path(args.manifest)
    manifest = load_json_file(manifest_path)
    release = str(args.release).upper()
    diagnostics_dir = manifest_path.parent / "diagnostics"
    output_path = (
        Path(args.output)
        if args.output
        else diagnostics_dir / f"landfire_{release.lower()}_checklist_support.json"
    )
    unsupported_output_path = (
        Path(args.unsupported_output)
        if args.unsupported_output
        else diagnostics_dir
        / f"landfire_{release.lower()}_unsupported_checklists.csv"
    )
    validation, unsupported = validate_landfire_checklist_support(
        plan,
        manifest,
        Path(args.checklists),
        release,
    )
    validation["manifest_path"] = str(manifest_path)
    write_landfire_checklist_support(
        validation,
        unsupported,
        output_path,
        unsupported_output_path,
    )
    print(
        f"LANDFIRE {release} checklist support: "
        f"{validation['eligible_checklist_count']:,}/"
        f"{validation['checklist_count']:,} checklists eligible"
    )
    for record in validation["support_by_radius"]:
        fraction = record["supported_fraction"]
        fraction_text = f"{fraction:.4%}" if fraction is not None else "n/a"
        print(
            f"  r{record['radius_m']}: "
            f"{record['supported_checklists']:,}/"
            f"{record['eligible_checklists']:,} supported "
            f"({fraction_text}); "
            f"{record['unsupported_checklists']:,} unsupported"
        )
    print(f"Wrote checklist-support validation to {output_path}")
    print(f"Wrote unsupported checklist details to {unsupported_output_path}")


def run_compare_landfire_releases(args: argparse.Namespace) -> None:
    plan = load_plan(Path(args.plan))
    manifest_path = Path(args.manifest)
    manifest = load_json_file(manifest_path)
    baseline_release = str(args.baseline_release).upper()
    comparison_release = str(args.comparison_release).upper()
    comparison_slug = (
        f"landfire_{baseline_release.lower()}_vs_"
        f"{comparison_release.lower()}_release_comparison"
    )
    diagnostics_dir = manifest_path.parent / "diagnostics"
    output_path = (
        Path(args.output)
        if args.output
        else diagnostics_dir / f"{comparison_slug}.json"
    )
    metrics_output_path = (
        Path(args.metrics_output)
        if args.metrics_output
        else diagnostics_dir / f"{comparison_slug}_metrics.csv"
    )
    summary, metrics = compare_landfire_releases(
        plan,
        manifest,
        baseline_release,
        comparison_release,
        tile_ids=args.tile_ids,
    )
    summary["manifest_path"] = str(manifest_path)
    write_landfire_release_comparison(
        summary,
        metrics,
        output_path,
        metrics_output_path,
    )
    print(
        f"Compared LANDFIRE {baseline_release} -> {comparison_release}: "
        f"{summary['release_tile_count']} release tiles structurally checked; "
        f"{summary['comparison_tile_count']} tiles pixel-compared; "
        f"{summary['matched_band_count']} matched bands"
    )
    print(
        "Maximum AOI-support difference: "
        f"{summary['maximum_absolute_support_difference']:.3g}; "
        f"nonzero tile/radius differences="
        f"{summary['nonzero_support_difference_count']}"
    )
    ranked = sorted(
        (
            metric
            for metric in metrics
            if metric["mean_absolute_delta"] is not None
        ),
        key=lambda value: value["mean_absolute_delta"],
        reverse=True,
    )
    print("Largest per-band mean absolute changes on compared tiles:")
    for metric in ranked[:8]:
        pearson = metric["pearson"]
        pearson_text = f"{pearson:.4f}" if pearson is not None else "n/a"
        print(
            f"  {metric['variable']} r{metric['radius_m']}: "
            f"MAE={metric['mean_absolute_delta']:.5f}, "
            f"mean delta={metric['mean_delta']:.5f}, "
            f"Pearson={pearson_text}, "
            f"mask mismatch={metric['mask_mismatch_cells']:,}"
        )
    print(f"Wrote release comparison to {output_path}")
    print(f"Wrote release metrics to {metrics_output_path}")


def run_derive_landfire_disturbance(args: argparse.Namespace) -> None:
    summary = derive_landfire_disturbance(
        plan=load_plan(Path(args.plan)),
        export_summary=load_json_file(Path(args.export_summary)),
        lookup_summary=load_json_file(Path(args.lookup_summary)),
        output_dir=Path(args.output_dir),
        neighborhoods_m=args.neighborhoods_m,
        minimum_coverage=args.minimum_coverage,
        overwrite=args.overwrite,
        write_vrt=not args.no_vrt,
        progress=True,
    )
    print(
        f"Derived LANDFIRE disturbance {summary['tile_id']}: "
        f"{summary['band_count']} bands, "
        f"{summary['derived_cog_count']} COGs, "
        f"{summary['derived_cog_bytes'] / (1024**2):.2f} MiB"
    )
    print(f"Wrote disturbance derivation summary to {summary['summary_path']}")


def run_validate_landfire_disturbance_derived(
    args: argparse.Namespace,
) -> None:
    plan = load_plan(Path(args.plan))
    summary_path = Path(args.summary)
    summary = load_json_file(summary_path)
    validation = validate_landfire_disturbance_derivation(plan, summary)
    diagnostics_dir = summary_path.parent / "diagnostics"
    output_path = (
        Path(args.output)
        if args.output
        else diagnostics_dir / "landfire_disturbance_validation.json"
    )
    write_landfire_disturbance_validation(validation, output_path)
    print(
        f"Validated LANDFIRE disturbance {validation['tile_id']}: "
        f"{validation['band_count']} bands, "
        f"{validation['derived_cog_count']} COGs, "
        f"{validation['empty_band_count']} empty logical bands"
    )
    for record in validation["band_statistics"]:
        support = record["supported_aoi_fraction"]
        support_text = f"{support:.2%}" if support is not None else "n/a"
        print(
            f"  {record['year']} r{record['radius_m']}: "
            f"support={support_text}, mean={record['mean']:.5f}, "
            f"max={record['maximum']:.5f}"
        )
    print(f"All checks passed: {validation['all_checks_passed']}")
    print(f"Wrote disturbance validation to {output_path}")
    if args.preview:
        preview_path = (
            Path(args.preview_output)
            if args.preview_output
            else diagnostics_dir
            / f"{validation['tile_id']}_landfire_disturbance_preview.png"
        )
        plot_landfire_disturbance_preview(plan, summary, preview_path)
        print(f"Wrote LANDFIRE disturbance preview to {preview_path}")
    if not validation["all_checks_passed"]:
        issue_summary = "; ".join(validation["issues"][:5])
        raise RuntimeError(
            f"Derived LANDFIRE disturbance validation failed with "
            f"{len(validation['issues'])} issue(s): {issue_summary}"
        )


def run_build_landfire_state(args: argparse.Namespace) -> None:
    result = build_landfire_state(
        plan=load_plan(Path(args.plan)),
        catalog=load_json_file(Path(args.catalog)),
        vegetation_crosswalk=load_json_file(
            Path(args.vegetation_crosswalk_summary)
        ),
        disturbance_lookup=load_json_file(
            Path(args.disturbance_lookup_summary)
        ),
        output_dir=Path(args.output_dir),
        profiles=args.profiles,
        tile_ids=args.tile_ids,
        releases=args.releases,
        components=args.components,
        buffer_m=args.buffer_m,
        minimum_coverage=args.minimum_coverage,
        minimum_lifeform_fraction=args.minimum_lifeform_fraction,
        timeout=args.timeout,
        max_units=args.max_units,
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
        overwrite=args.overwrite,
        progress=True,
    )
    manifest = result["manifest"]
    expected = manifest["expected_state"]
    mode = "dry run" if result["dry_run"] else "build"
    print(
        f"LANDFIRE state {mode}: {manifest['completed_unit_count']}/"
        f"{manifest['unit_count']} units complete; "
        f"resumed manifest={result['resumed_manifest']}"
    )
    print(
        f"Expected shared raster bands: "
        f"{expected['shared_raster_band_count']}"
    )
    for profile, count in expected["profile_raster_band_counts"].items():
        print(f"  {profile}: {count} raster bands")
    for profile, output in manifest.get("profile_outputs", {}).items():
        print(f"  {profile} VRT: {output['logical_vrt']}")
    print(f"Wrote state manifest to {result['manifest_path']}")


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
    if args.command == "register-nlcd-local":
        run_register_nlcd_local(args)
        return
    if args.command == "register-nlcd-aws":
        run_register_nlcd_aws(args)
        return
    if args.command == "validate-nlcd-sources":
        run_validate_nlcd_sources(args)
        return
    if args.command == "derive-nlcd":
        run_derive_nlcd(args)
        return
    if args.command == "validate-nlcd-derived":
        run_validate_nlcd_derived(args)
        return
    if args.command == "validate-nlcd-checklist-support":
        run_validate_nlcd_checklist_support(args)
        return
    if args.command == "catalog-landfire":
        run_catalog_landfire(args)
        return
    if args.command == "catalog-landfire-attributes":
        run_catalog_landfire_attributes(args)
        return
    if args.command == "build-landfire-crosswalks":
        run_build_landfire_crosswalks(args)
        return
    if args.command == "build-landfire-disturbance-lookup":
        run_build_landfire_disturbance_lookup(args)
        return
    if args.command == "export-landfire":
        run_export_landfire(args)
        return
    if args.command == "derive-landfire":
        run_derive_landfire(args)
        return
    if args.command == "validate-landfire-derived":
        run_validate_landfire_derived(args)
        return
    if args.command == "validate-landfire-checklist-support":
        run_validate_landfire_checklist_support(args)
        return
    if args.command == "compare-landfire-releases":
        run_compare_landfire_releases(args)
        return
    if args.command == "derive-landfire-disturbance":
        run_derive_landfire_disturbance(args)
        return
    if args.command == "validate-landfire-disturbance-derived":
        run_validate_landfire_disturbance_derived(args)
        return
    if args.command == "build-landfire-state":
        run_build_landfire_state(args)
        return
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
