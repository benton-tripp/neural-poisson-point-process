"""Resolve release-pinned LANDFIRE layers from official public services."""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PRODUCTS_API_URL = "https://lfps.usgs.gov/api/products"
IMAGE_SERVER_ROOT = "https://lfps.usgs.gov/arcgis/rest/services"
FULL_EXTENT_ROOT = "https://www.landfire.gov/data-downloads"
EXPECTED_WKID = 5070
EXPECTED_RESOLUTION_M = 30.0
DEFAULT_VEGETATION_RELEASES = ["LF2016", "LF2022", "LF2023"]
VEGETATION_PRODUCTS = {
    "EVT": "Existing Vegetation Type",
    "EVC": "Existing Vegetation Cover",
    "EVH": "Existing Vegetation Height",
}
SUPPORTED_PRODUCTS = {*VEGETATION_PRODUCTS, "Dist"}
FINAL_DISTURBANCE_NAME = "Final Annual Disturbance"


def normalize_release(value: Any) -> str:
    text = str(value).strip().upper()
    if text.isdigit() and len(text) == 4:
        text = f"LF{text}"
    if not re.fullmatch(r"LF\d{4}", text):
        raise ValueError(f"Invalid LANDFIRE release {value!r}; expected LFYYYY.")
    return text


def release_year(release: str) -> int:
    return int(normalize_release(release)[2:])


def validate_years(years: list[int]) -> list[int]:
    if not years or any(not isinstance(year, int) for year in years):
        raise ValueError("LANDFIRE observation years must be a non-empty integer list.")
    if len(set(years)) != len(years):
        raise ValueError("LANDFIRE observation years must not contain duplicates.")
    return sorted(years)


def validate_products(products: list[str]) -> list[str]:
    if not products:
        raise ValueError("At least one LANDFIRE product is required.")
    unknown = sorted(set(products) - SUPPORTED_PRODUCTS)
    if unknown:
        raise ValueError(
            f"Unknown LANDFIRE products: {', '.join(unknown)}. Choose from "
            f"{', '.join(sorted(SUPPORTED_PRODUCTS))}."
        )
    return list(dict.fromkeys(products))


def validate_releases(releases: list[Any]) -> list[str]:
    normalized = [normalize_release(value) for value in releases]
    if not normalized:
        raise ValueError("At least one LANDFIRE vegetation release is required.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("LANDFIRE vegetation releases must not contain duplicates.")
    return sorted(normalized, key=release_year)


def latest_release_by_year(
    observation_years: list[int], releases: list[Any]
) -> dict[int, str]:
    years = validate_years(observation_years)
    normalized = validate_releases(releases)
    mapping: dict[int, str] = {}
    for year in years:
        candidates = [release for release in normalized if release_year(release) <= year]
        if not candidates:
            raise ValueError(
                f"No LANDFIRE vegetation release is available on or before {year}."
            )
        mapping[year] = candidates[-1]
    return mapping


def validate_release_by_year(
    observation_years: list[int],
    releases: list[Any],
    release_by_year: dict[Any, Any] | None,
) -> dict[int, str]:
    years = validate_years(observation_years)
    normalized = validate_releases(releases)
    if release_by_year is None:
        return latest_release_by_year(years, normalized)
    mapping = {int(year): normalize_release(value) for year, value in release_by_year.items()}
    missing = sorted(set(years) - set(mapping))
    extra = sorted(set(mapping) - set(years))
    if missing or extra:
        raise ValueError(
            "LANDFIRE release_by_year must exactly match observation years; "
            f"missing={missing}, extra={extra}."
        )
    for year, release in mapping.items():
        if release not in normalized:
            raise ValueError(f"LANDFIRE release {release} for {year} is not requested.")
        if release_year(release) > year:
            raise ValueError(
                f"LANDFIRE release {release} is later than observation year {year}."
            )
    return dict(sorted(mapping.items()))


def request_json(
    url: str,
    timeout: float,
    session: requests.Session | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    requester = session or requests
    response = requester.get(
        url,
        params=params,
        timeout=timeout,
        headers={"User-Agent": "ebird-covariate-pipeline/1.0"},
    )
    response.raise_for_status()
    return response.json()


def fetch_products(
    timeout: float,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    payload = request_json(PRODUCTS_API_URL, timeout, session=session)
    products = payload.get("products") if isinstance(payload, dict) else None
    if not isinstance(products, list) or not products:
        raise ValueError("The LFPS products endpoint returned no products.")
    required_fields = {"productName", "layerName", "acronym", "version", "conus"}
    for index, product in enumerate(products):
        if not isinstance(product, dict) or not required_fields <= set(product):
            raise ValueError(f"LFPS product record {index} is missing required fields.")
    return products


def require_one_product(
    products: list[dict[str, Any]],
    *,
    version: str | None = None,
    acronym: str,
    product_name: str,
    layer_suffix: str | None = None,
) -> dict[str, Any]:
    matches = [
        product
        for product in products
        if product["acronym"] == acronym
        and product["productName"] == product_name
        and product["conus"] is True
        and (version is None or product["version"] == version)
        and (layer_suffix is None or product["layerName"].endswith(layer_suffix))
    ]
    if len(matches) != 1:
        label = f"{version or '*'}:{acronym}:{layer_suffix or '*'}"
        raise ValueError(
            f"Expected exactly one CONUS LFPS product for {label}; found {len(matches)}."
        )
    return matches[0]


def image_server_url(product: dict[str, Any]) -> str:
    folder = (
        "Landfire_Disturbance"
        if product["acronym"] == "Dist"
        else f"Landfire_{product['version']}"
    )
    return (
        f"{IMAGE_SERVER_ROOT}/{folder}/"
        f"{product['layerName']}_CONUS/ImageServer"
    )


def full_extent_download_url(product: dict[str, Any]) -> str | None:
    if product["acronym"] not in VEGETATION_PRODUCTS:
        return None
    return (
        f"{FULL_EXTENT_ROOT}/CONUS_{product['version']}/"
        f"{product['layerName']}_CONUS.zip"
    )


def inspect_image_service(
    product: dict[str, Any],
    timeout: float,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    url = image_server_url(product)
    metadata = request_json(url, timeout, session=session, params={"f": "pjson"})
    if not isinstance(metadata, dict) or metadata.get("error"):
        raise ValueError(f"LANDFIRE ImageServer returned an error for {url}.")
    wkid = (metadata.get("spatialReference") or {}).get("latestWkid") or (
        metadata.get("spatialReference") or {}
    ).get("wkid")
    checks = {
        "single_band": metadata.get("bandCount") == 1,
        "projected_in_epsg_5070": wkid == EXPECTED_WKID,
        "native_resolution_30m": (
            float(metadata.get("pixelSizeX", -1)) == EXPECTED_RESOLUTION_M
            and float(metadata.get("pixelSizeY", -1)) == EXPECTED_RESOLUTION_M
        ),
        "thematic_service": (
            metadata.get("serviceDataType") == "esriImageServiceDataTypeThematic"
        ),
        "nearest_neighbor_default": metadata.get("defaultResamplingMethod") == "Nearest",
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(
            f"LANDFIRE ImageServer metadata failed checks for {url}: "
            + ", ".join(failed)
        )
    return {
        "image_server_url": url,
        "service_name": metadata.get("name"),
        "wkid": wkid,
        "pixel_size_m": [metadata.get("pixelSizeX"), metadata.get("pixelSizeY")],
        "pixel_type": metadata.get("pixelType"),
        "band_count": metadata.get("bandCount"),
        "nodata": metadata.get("noDataValue"),
        "extent": metadata.get("extent"),
        "default_resampling_method": metadata.get("defaultResamplingMethod"),
        "max_image_width": metadata.get("maxImageWidth"),
        "max_image_height": metadata.get("maxImageHeight"),
        "checks": checks,
    }


def _api_snapshot_sha256(products: list[dict[str, Any]]) -> str:
    payload = json.dumps(products, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_catalog(
    observation_years: list[int],
    vegetation_releases: list[Any] | None = None,
    release_by_year: dict[Any, Any] | None = None,
    disturbance_years: list[int] | None = None,
    products: list[str] | None = None,
    timeout: float = 60.0,
    max_workers: int = 6,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    years = validate_years(observation_years)
    releases = validate_releases(
        vegetation_releases or DEFAULT_VEGETATION_RELEASES
    )
    selected_products = validate_products(
        products or [*VEGETATION_PRODUCTS, "Dist"]
    )
    mapping = validate_release_by_year(years, releases, release_by_year)
    disturbance = validate_years(disturbance_years or years)
    if not set(disturbance) <= set(years):
        raise ValueError("LANDFIRE disturbance years must be observation years.")

    api_products = fetch_products(timeout, session=session)
    selected: list[dict[str, Any]] = []
    for release in releases:
        for acronym in VEGETATION_PRODUCTS:
            if acronym not in selected_products:
                continue
            product = require_one_product(
                api_products,
                version=release,
                acronym=acronym,
                product_name=VEGETATION_PRODUCTS[acronym],
            )
            selected.append(
                {
                    **product,
                    "role": "vegetation_release",
                    "content_year": release_year(release),
                    "observation_year": None,
                }
            )
    if "Dist" in selected_products:
        for year in disturbance:
            product = require_one_product(
                api_products,
                acronym="Dist",
                product_name=FINAL_DISTURBANCE_NAME,
                layer_suffix=f"Dist{year % 100:02d}",
            )
            selected.append(
                {
                    **product,
                    "role": "annual_disturbance",
                    "content_year": year,
                    "observation_year": year,
                }
            )

    worker_count = max(1, min(max_workers, len(selected)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        services = list(
            executor.map(
                lambda product: inspect_image_service(
                    product, timeout, session=session
                ),
                selected,
            )
        )
    layers = [
        {
            **product,
            **service,
            "full_extent_download_url": full_extent_download_url(product),
        }
        for product, service in zip(selected, services, strict=True)
    ]

    available_vegetation_versions = sorted(
        {
            product["version"]
            for product in api_products
            if product["conus"] is True
            and product["acronym"] in VEGETATION_PRODUCTS
        },
        key=release_year,
    )
    return {
        "schema_version": 1,
        "resolved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": "landfire",
        "products_api_url": PRODUCTS_API_URL,
        "products_api_record_count": len(api_products),
        "products_api_snapshot_sha256": _api_snapshot_sha256(api_products),
        "observation_years": years,
        "products": selected_products,
        "vegetation_releases": releases,
        "vegetation_release_by_year": {
            str(year): release for year, release in mapping.items()
        },
        "vegetation_source_age_by_year": {
            str(year): year - release_year(release)
            for year, release in mapping.items()
        },
        "available_lfps_vegetation_versions": available_vegetation_versions,
        "disturbance_years": disturbance if "Dist" in selected_products else [],
        "layers": layers,
        "layer_count": len(layers),
        "all_services_validated": all(
            all(layer["checks"].values()) for layer in layers
        ),
        "acquisition_status": "official_public_imageservers_resolved",
        "notes": [
            "Catalog resolution downloads metadata only, not raster values.",
            "Vegetation releases are periodic; observation years select the latest requested non-future release.",
            "LF2020 vegetation is archived and absent from the current LFPS catalog, so the NC pilot uses LF2016 for 2020-2021 and records source age.",
            "Annual disturbance layers use their exact content years even when distributed under a later LANDFIRE version.",
            "Derived categorical rasters must use nearest-neighbor source reads before class-fraction aggregation.",
        ],
    }


def write_catalog(catalog: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
