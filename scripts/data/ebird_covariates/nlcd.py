"""Annual NLCD Collection 1.2 ScienceBase metadata resolution."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


SCIENCEBASE_ROOT_ITEM = "655ceb8ad34ee4b6e05cc51a"
SCIENCEBASE_ITEM_URL = "https://www.sciencebase.gov/catalog/item/{item_id}?format=json"
COLLECTION = 1
VERSION = 2
REGION = "CU"
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
        "size_bytes": int(file["size"]),
        "acquisition_status": ACQUISITION_STATUS,
        "direct_download_available": False,
        **link_fields,
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
