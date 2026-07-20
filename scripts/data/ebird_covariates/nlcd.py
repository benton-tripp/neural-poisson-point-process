"""Annual NLCD Collection 1.2 ScienceBase metadata resolution."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import rasterio
import requests


SCIENCEBASE_ROOT_ITEM = "655ceb8ad34ee4b6e05cc51a"
SCIENCEBASE_ITEM_URL = "https://www.sciencebase.gov/catalog/item/{item_id}?format=json"
COLLECTION = 1
VERSION = 2
REGION = "CU"
AWS_BUCKET = "usgs-landcover"
AWS_REGION = "us-west-2"
ACQUISITION_STATUS = "metadata_resolved_manual_or_aws_credentials_required"

PRODUCTS = {
    "LndCov": {
        "name": "Land Cover",
        "item_id": "697b9279b66b0197c3043cc3",
    },
    "LndChg": {
        "name": "Land Cover Change",
        "item_id": "697b9298b66b0197c3043cc5",
    },
    "LndCnf": {
        "name": "Land Cover Confidence",
        "item_id": "697b92b8b66b0197c3043cc7",
    },
    "FctImp": {
        "name": "Fractional Impervious Surface",
        "item_id": "697b907eb66b0197c3043c9f",
    },
    "ImpDsc": {
        "name": "Impervious Descriptor",
        "item_id": "697b925eb66b0197c3043cbf",
    },
    "SpcChg": {
        "name": "Spectral Change Day of Year",
        "item_id": "697b92d3b66b0197c3043cc9",
    },
}

LAND_COVER_CLASSES = {
    11: "open_water",
    12: "perennial_ice_snow",
    21: "developed_open_space",
    22: "developed_low_intensity",
    23: "developed_medium_intensity",
    24: "developed_high_intensity",
    31: "barren_land",
    41: "deciduous_forest",
    42: "evergreen_forest",
    43: "mixed_forest",
    52: "shrub_scrub",
    71: "grassland_herbaceous",
    81: "pasture_hay",
    82: "cultivated_crops",
    90: "woody_wetlands",
    95: "emergent_herbaceous_wetlands",
}


def validate_products(products: list[str]) -> list[str]:
    if not products:
        raise ValueError("At least one Annual NLCD product is required.")
    unknown = sorted(set(products) - set(PRODUCTS))
    if unknown:
        raise ValueError(
            f"Unknown Annual NLCD product codes: {', '.join(unknown)}. "
            f"Choose from {', '.join(PRODUCTS)}."
        )
    return list(dict.fromkeys(products))


def validate_years(years: list[int]) -> list[int]:
    if not years or any(not isinstance(year, int) for year in years):
        raise ValueError("Annual NLCD years must be a non-empty integer list.")
    if len(set(years)) != len(years):
        raise ValueError("Annual NLCD years must not contain duplicates.")
    if min(years) < 1985 or max(years) > 2025:
        raise ValueError("Annual NLCD Collection 1.2 supports years 1985-2025.")
    return sorted(years)


def fetch_item_metadata(
    product_code: str,
    timeout: float,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    product = PRODUCTS[product_code]
    requester = session or requests
    response = requester.get(
        SCIENCEBASE_ITEM_URL.format(item_id=product["item_id"]),
        timeout=timeout,
        headers={"User-Agent": "ebird-covariate-pipeline/1.0"},
    )
    response.raise_for_status()
    item = response.json()
    expected_title_fragment = f"Collection 1.2 {product['name']}"
    if expected_title_fragment not in item.get("title", ""):
        raise ValueError(
            f"ScienceBase item {product['item_id']} title did not match "
            f"{expected_title_fragment!r}: {item.get('title')!r}"
        )
    return item


def expected_filename(product_code: str, year: int) -> str:
    return (
        f"Annual_NLCD_{product_code}_{year}_{REGION}_"
        f"C{COLLECTION}V{VERSION}.zip"
    )


def expected_raster_filename(product_code: str, year: int) -> str:
    return (
        f"Annual_NLCD_{product_code}_{year}_{REGION}_"
        f"C{COLLECTION}V{VERSION}.tif"
    )


def aws_mosaic_key(product_code: str, year: int) -> str:
    return (
        f"annual-nlcd/c{COLLECTION}/v{VERSION}/{REGION.lower()}/mosaic/"
        f"{expected_raster_filename(product_code, year)}"
    )


def aws_mosaic_uri(product_code: str, year: int) -> str:
    return f"s3://{AWS_BUCKET}/{aws_mosaic_key(product_code, year)}"


def gdal_vsis3_uri(product_code: str, year: int) -> str:
    return f"/vsis3/{AWS_BUCKET}/{aws_mosaic_key(product_code, year)}"


def file_record(item: dict[str, Any], product_code: str, year: int) -> dict[str, Any]:
    filename = expected_filename(product_code, year)
    matches = [file for file in item.get("files", []) if file.get("name") == filename]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {filename} on ScienceBase item {item.get('id')}; "
            f"found {len(matches)}."
        )
    file = matches[0]
    link_fields = {
        "sciencebase_file_manager_url": file.get("url"),
        "sciencebase_download_uri": file.get("downloadUri"),
        "sciencebase_download_request_url": file.get("s3DownloadRequestPageUri"),
    }
    for field_name, value in link_fields.items():
        if value is not None and (
            not isinstance(value, str) or not value.startswith("https://")
        ):
            raise ValueError(
                f"ScienceBase file {filename} has an invalid {field_name}."
            )
    if not any(link_fields.values()):
        raise ValueError(f"ScienceBase file {filename} has no HTTPS metadata links.")
    checksum = file.get("checksum") or {}
    return {
        "product_code": product_code,
        "product_name": PRODUCTS[product_code]["name"],
        "year": year,
        "filename": filename,
        "raster_filename": expected_raster_filename(product_code, year),
        "size_bytes": int(file["size"]),
        "acquisition_status": ACQUISITION_STATUS,
        "direct_download_available": False,
        **link_fields,
        "aws_s3_uri": aws_mosaic_uri(product_code, year),
        "aws_gdal_vsi_uri": gdal_vsis3_uri(product_code, year),
        "aws_region": AWS_REGION,
        "aws_requester_pays": True,
        "remote_checksum": checksum.get("value"),
        "remote_checksum_type": checksum.get("type"),
        "sciencebase_item_id": item["id"],
        "sciencebase_item_title": item["title"],
    }


def resolve_catalog(
    years: list[int],
    products: list[str],
    extra_years_by_product: dict[str, list[int]] | None = None,
    timeout: float = 120.0,
    max_workers: int = 4,
) -> dict[str, Any]:
    years = validate_years(years)
    products = validate_products(products)
    extra_years_by_product = extra_years_by_product or {}
    unknown_extra_products = sorted(set(extra_years_by_product) - set(products))
    if unknown_extra_products:
        raise ValueError(
            "Extra Annual NLCD years reference unrequested products: "
            + ", ".join(unknown_extra_products)
        )
    product_years = {
        product: sorted(
            set(years)
            | set(validate_years(extra_years_by_product.get(product, years)))
        )
        for product in products
    }
    worker_count = max(1, min(max_workers, len(products)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        metadata_values = list(
            executor.map(
                lambda code: fetch_item_metadata(code, timeout),
                products,
            )
        )
    metadata = dict(zip(products, metadata_values, strict=True))
    files = [
        file_record(metadata[product], product, year)
        for product in products
        for year in product_years[product]
    ]
    return {
        "schema_version": 2,
        "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": "annual_nlcd",
        "sciencebase_root_item_id": SCIENCEBASE_ROOT_ITEM,
        "collection": COLLECTION,
        "version": VERSION,
        "release": f"C{COLLECTION}V{VERSION}",
        "region": REGION,
        "years": years,
        "product_years": product_years,
        "products": products,
        "files": files,
        "acquisition_status": ACQUISITION_STATUS,
        "total_source_bytes": sum(file["size_bytes"] for file in files),
        "land_cover_classes": [
            {"value": value, "name": name}
            for value, name in LAND_COVER_CLASSES.items()
        ],
        "notes": [
            "ScienceBase metadata is resolved dynamically; the Collection 1.2 item ids are pinned in code.",
            "ScienceBase large-file links are request or file-manager pages, not unattended public download URLs.",
            "Automated acquisition requires an authenticated requester-pays AWS path or a separately resolved official MRLC download.",
            "Large ScienceBase files do not currently expose checksums in item metadata; acquired archives must receive a local SHA-256.",
            "Catalog resolution downloads metadata only, not raster zip files.",
        ],
    }


def write_catalog(catalog: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gdal_vsizip_uri(archive_path: Path, member: str) -> str:
    archive = archive_path.resolve().as_posix()
    return f"/vsizip/{archive}/{member.lstrip('/')}"


def inspect_raster(raster_uri: str) -> dict[str, Any]:
    with rasterio.open(raster_uri) as dataset:
        if dataset.crs is None:
            raise ValueError(f"Annual NLCD raster has no CRS: {raster_uri}")
        return {
            "crs": dataset.crs.to_string(),
            "width": dataset.width,
            "height": dataset.height,
            "count": dataset.count,
            "dtype": dataset.dtypes[0],
            "nodata": dataset.nodata,
            "bounds": list(dataset.bounds),
            "transform": list(dataset.transform)[:6],
        }


def index_local_files(input_dir: Path) -> dict[str, Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Annual NLCD input directory does not exist: {input_dir}")
    indexed: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in input_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in indexed:
            duplicates.add(path.name)
        else:
            indexed[path.name] = path
    if duplicates:
        raise ValueError(
            "Annual NLCD input directory contains duplicate filenames: "
            + ", ".join(sorted(duplicates))
        )
    return indexed


def resolve_local_raster(
    catalog_file: dict[str, Any],
    indexed_files: dict[str, Path],
    calculate_sha256: bool,
) -> dict[str, Any]:
    raster_filename = catalog_file["raster_filename"]
    archive_filename = catalog_file["filename"]
    raster_path = indexed_files.get(raster_filename)
    archive_path = indexed_files.get(archive_filename)
    if raster_path is not None:
        source_path = raster_path.resolve()
        raster_uri = str(source_path)
        source_type = "local_tiff"
        member = None
        catalog_size_matches = None
    elif archive_path is not None:
        source_path = archive_path.resolve()
        with zipfile.ZipFile(source_path) as archive:
            matches = [
                name
                for name in archive.namelist()
                if Path(name).name == raster_filename
            ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one {raster_filename} in {source_path}; "
                f"found {len(matches)}."
            )
        member = matches[0]
        raster_uri = gdal_vsizip_uri(source_path, member)
        source_type = "local_zip"
        catalog_size_matches = source_path.stat().st_size == catalog_file["size_bytes"]
    else:
        raise FileNotFoundError(
            f"Missing {raster_filename} or {archive_filename} under the input directory."
        )

    raster_metadata = inspect_raster(raster_uri)
    if raster_metadata["count"] != 1:
        raise ValueError(f"Expected one raster band in {raster_uri}.")
    return {
        "product_code": catalog_file["product_code"],
        "product_name": catalog_file["product_name"],
        "year": catalog_file["year"],
        "source_type": source_type,
        "source_path": str(source_path),
        "source_size_bytes": source_path.stat().st_size,
        "source_sha256": sha256_file(source_path) if calculate_sha256 else None,
        "archive_member": member,
        "raster_uri": raster_uri,
        "catalog_archive_filename": archive_filename,
        "catalog_archive_size_bytes": catalog_file["size_bytes"],
        "catalog_archive_size_matches": catalog_size_matches,
        "raster": raster_metadata,
    }


def register_local_sources(
    catalog: dict[str, Any],
    input_dir: Path,
    calculate_sha256: bool = False,
) -> dict[str, Any]:
    indexed_files = index_local_files(input_dir)
    sources = [
        resolve_local_raster(file, indexed_files, calculate_sha256)
        for file in catalog["files"]
    ]
    return {
        "schema_version": 1,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": "annual_nlcd",
        "release": catalog["release"],
        "backend": "local",
        "input_dir": str(input_dir.resolve()),
        "sha256_calculated": calculate_sha256,
        "sources": sources,
    }


def register_aws_sources(catalog: dict[str, Any]) -> dict[str, Any]:
    sources = [
        {
            "product_code": file["product_code"],
            "product_name": file["product_name"],
            "year": file["year"],
            "source_type": "aws_requester_pays_cog",
            "source_path": file["aws_s3_uri"],
            "raster_uri": file["aws_gdal_vsi_uri"],
            "aws_region": file["aws_region"],
            "aws_requester_pays": True,
            "validated": False,
        }
        for file in catalog["files"]
    ]
    return {
        "schema_version": 1,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": "annual_nlcd",
        "release": catalog["release"],
        "backend": "aws_requester_pays",
        "aws_bucket": AWS_BUCKET,
        "aws_region": AWS_REGION,
        "sources": sources,
    }


def write_source_registration(registration: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(registration, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def validate_registered_sources(registration: dict[str, Any]) -> dict[str, Any]:
    """Open registered rasters and validate their basic and temporal grid contract."""
    if registration.get("source_id") != "annual_nlcd":
        raise ValueError("Source registration must describe annual_nlcd.")
    sources = registration.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("Annual NLCD source registration has no sources.")

    environment = (
        rasterio.Env(
            AWS_REQUEST_PAYER="requester",
            AWS_REGION=registration.get("aws_region", AWS_REGION),
        )
        if registration.get("backend") == "aws_requester_pays"
        else nullcontext()
    )
    validated: list[dict[str, Any]] = []
    with environment:
        for source in sources:
            try:
                metadata = inspect_raster(source["raster_uri"])
            except Exception as exc:
                product = source.get("product_code", "unknown")
                year = source.get("year", "unknown")
                raise RuntimeError(
                    f"Could not open registered Annual NLCD source {product}:{year} "
                    f"at {source.get('raster_uri')}. For requester-pays AWS sources, "
                    "configure standard AWS credentials with permission to read the bucket."
                ) from exc
            if metadata["count"] != 1:
                raise ValueError(
                    f"Annual NLCD source {source['product_code']}:{source['year']} "
                    "must contain exactly one raster band."
                )
            validated.append(
                {
                    "product_code": source["product_code"],
                    "year": source["year"],
                    "raster_uri": source["raster_uri"],
                    "raster": metadata,
                }
            )

    land_cover = sorted(
        (record for record in validated if record["product_code"] == "LndCov"),
        key=lambda record: record["year"],
    )
    if land_cover:
        reference = land_cover[0]["raster"]
        grid_fields = ("crs", "width", "height", "transform")
        for record in land_cover[1:]:
            if any(record["raster"][field] != reference[field] for field in grid_fields):
                raise ValueError(
                    "Annual NLCD land-cover rasters are not on one aligned grid: "
                    f"{land_cover[0]['year']} versus {record['year']}."
                )

    return {
        "schema_version": 1,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": "annual_nlcd",
        "release": registration.get("release"),
        "backend": registration.get("backend"),
        "source_count": len(validated),
        "all_sources_opened": True,
        "land_cover_grids_aligned": bool(land_cover),
        "sources": validated,
    }


def write_source_validation(validation: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)


def parse_year_expression(value: str) -> list[int]:
    years: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        match = re.fullmatch(r"(\d{4})-(\d{4})", token)
        if match:
            start, end = (int(item) for item in match.groups())
            if start > end:
                raise ValueError(f"Invalid descending year range: {token}")
            years.update(range(start, end + 1))
        elif re.fullmatch(r"\d{4}", token):
            years.add(int(token))
        else:
            raise ValueError(f"Invalid year token: {token}")
    return validate_years(sorted(years))
