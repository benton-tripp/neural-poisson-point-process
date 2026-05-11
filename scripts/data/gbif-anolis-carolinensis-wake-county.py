"""
Download Green Anole (Anolis carolinensis) GBIF occurrence records
from Wake County, NC, 2015–2025.

Records are first queried from GBIF using a bounding box, then masked
to the Wake County boundary before saving.
"""

import os
import time
import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
import matplotlib.pyplot as plt

os.makedirs("data", exist_ok=True)

GBIF_OCCURRENCE_URL = "https://api.gbif.org/v1/occurrence/search"

# Bounding box coordinates:
# xmin, ymin = western longitude, southern latitude
# xmax, ymax = eastern longitude, northern latitude
XMIN, YMIN = -78.99507, 35.51948
XMAX, YMAX = -78.25368, 36.07629

BBOX_WKT = (
    f"POLYGON(("
    f"{XMIN} {YMIN},"
    f"{XMAX} {YMIN},"
    f"{XMAX} {YMAX},"
    f"{XMIN} {YMAX},"
    f"{XMIN} {YMIN}"
    f"))"
)

PARAMS = {
    "taxon_key": 2466939,              # Green Anole, Anolis carolinensis
    "country": "US",
    "continent": "NORTH_AMERICA",
    "year": "2015,2025",
    "geometry": BBOX_WKT,
    "has_coordinate": "true",
    "has_geospatial_issue": "false",
    "occurrence_status": "PRESENT",
    "limit": 300,
    "offset": 0,
}

# Wake County boundary
WAKE_COUNTY_URL = (
    "https://services1.arcgis.com/a7CWfuGP5ZnLYE7I/arcgis/rest/services/"
    "CountyLine/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&outSR=4326&f=json"
)

OUTPUT_CSV = "data/green_anole_gbif_wake_county_2015_2025.csv"


def fetch_gbif_occurrences(params, max_records=None, pause=0.25):
    """
    Fetch paginated GBIF occurrence records.
    """
    records = []
    offset = 0
    limit = params.get("limit", 300)

    while True:
        query = params.copy()
        query["offset"] = offset
        query["limit"] = limit

        response = requests.get(GBIF_OCCURRENCE_URL, params=query, timeout=60)
        response.raise_for_status()
        data = response.json()

        batch = data.get("results", [])
        if not batch:
            break

        records.extend(batch)

        print(f"Fetched {len(records):,} of {data.get('count', 'unknown'):,} records")

        if data.get("endOfRecords", False):
            break

        offset += limit

        if max_records is not None and len(records) >= max_records:
            records = records[:max_records]
            break

        time.sleep(pause)

    return pd.DataFrame(records)


def fetch_wake_county_boundary(url):
    """
    Fetch Wake County boundary from ArcGIS REST JSON and return a GeoDataFrame.

    Assumes ArcGIS geometry with polygon rings in EPSG:4326.
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()

    polygons = []

    for feature in data.get("features", []):
        geometry = feature.get("geometry", {})
        rings = geometry.get("rings", [])

        for ring in rings:
            poly = Polygon(ring)
            if poly.is_valid and not poly.is_empty:
                polygons.append(poly)

    if not polygons:
        raise ValueError("No valid Wake County polygon rings found.")

    boundary_geom = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]

    return gpd.GeoDataFrame(
        {"name": ["Wake County"]},
        geometry=[boundary_geom],
        crs="EPSG:4326",
    )


def mask_occurrences_to_boundary(df, boundary_gdf):
    """
    Convert GBIF occurrence records to point geometries and retain only
    records within the county boundary.
    """
    required_cols = {"decimalLongitude", "decimalLatitude"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required coordinate columns: {missing}")

    df = df.dropna(subset=["decimalLongitude", "decimalLatitude"]).copy()

    points_gdf = gpd.GeoDataFrame(
        df,
        geometry=[
            Point(xy)
            for xy in zip(df["decimalLongitude"], df["decimalLatitude"])
        ],
        crs="EPSG:4326",
    )

    # Ensure both layers use the same CRS
    boundary_gdf = boundary_gdf.to_crs(points_gdf.crs)

    masked = gpd.sjoin(
        points_gdf,
        boundary_gdf[["geometry"]],
        how="inner",
        predicate="within",
    )

    masked = masked.drop(columns=["index_right"])

    return masked


df = fetch_gbif_occurrences(PARAMS)

wake_boundary = fetch_wake_county_boundary(WAKE_COUNTY_URL)
df_masked = mask_occurrences_to_boundary(df, wake_boundary)

# Optional: keep a practical subset of columns if present
cols_to_keep = [
    "key",
    "scientificName",
    "species",
    "decimalLatitude",
    "decimalLongitude",
    "eventDate",
    "year",
    "countryCode",
    "stateProvince",
    "locality",
    "basisOfRecord",
    "datasetName",
    "institutionCode",
    "catalogNumber",
    "recordedBy",
    "identifiedBy",
    "coordinateUncertaintyInMeters",
    "license",
    "references",
]

existing_cols = [c for c in cols_to_keep if c in df_masked.columns]
df_out = df_masked[existing_cols].copy()

df_out.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved {len(df_out):,} Wake County records to {OUTPUT_CSV}")
print(df_out.head())


# Plot Wake County boundary with GBIF occurrence points overlaid
fig, ax = plt.subplots(figsize=(8, 8))

wake_boundary.boundary.plot(
    ax=ax,
    linewidth=1.5
)

df_masked.plot(
    ax=ax,
    markersize=20,
    alpha=0.7
)

ax.set_title("Green Anole GBIF Observations in Wake County, NC (2015–2025)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_aspect("equal")

plt.tight_layout()
# plt.show()
fig.savefig(
    "images/green_anole_gbif_wake_county_map.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)