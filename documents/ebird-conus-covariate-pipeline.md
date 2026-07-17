# eBird CONUS Covariate Pipeline

## Status

- Started: 2026-07-17
- Current phase: Phase 3 in progress; Annual NLCD catalog adapter complete,
  raster acquisition/derivation pending
- Initial study area: North Carolina
- Intended extent: conterminous United States (CONUS)
- Primary consumer: locality-season availability/detection models
- Related analysis ledger:
  [ebird-joint-effort-gnn-methods.md](ebird-joint-effort-gnn-methods.md)

This document is the design specification and implementation ledger for a
versioned environmental and observer-access covariate pipeline. It should retain
completed decisions, source versions, negative findings, validation results,
and exact commands as the pipeline develops.

## Purpose

Build a reproducible U.S. covariate system that can enrich eBird checklists and
locality-season groups without relying on North Carolina-specific habitat
labels. The pipeline must support:

- ecological availability modeling
- a separate observation/access process
- date-matched yearly and monthly predictors
- directional transfer tests across regions and observer regimes
- later availability-side GNN experiments

The immediate product will be an NC sample used to rerun the promoted
repeated-visit model with the likelihood, split, support rules, and optimizer
held fixed.

## Goals

1. Use authoritative, broadly available U.S. source datasets.
2. Normalize all modeling products to a stable CONUS metric grid.
3. Preserve temporal cadence instead of collapsing changing sources into one
   timeless average.
4. Summarize daily climate monthly and seasonally while retaining observation
   year.
5. Derive multi-scale habitat context rather than sampling only the raster cell
   beneath a checklist.
6. Keep ecological and observer-access bands distinguishable through explicit
   channel metadata.
7. Record source version, source date, coverage, transformation, units,
   resampling, and checksums.
8. Produce one logical layered raster for each build, with a materialized
   multi-band GeoTIFF available for regional samples such as NC.
9. Support resumable, tiled processing so one failed source or tile does not
   invalidate the whole build.
10. Make the same feature definitions reusable in other states and CONUS.

## Non-Goals

- Do not create a hand-labeled NC habitat map.
- Do not treat observer accessibility as ecological suitability.
- Do not overwrite the current four-covariate NC dataset.
- Do not force every source onto an artificial annual cadence.
- Do not infer historical change from a newer map replacing an older mapping
  project unless the source explicitly supports change analysis.
- Do not tune the latent model while evaluating the first enriched dataset.
- Do not make a monolithic physical CONUS GeoTIFF the only source of truth.

## Core Output Decision

### One logical raster, tiled physical storage

The pipeline will expose a single logical multi-band raster:

```text
data/ebird/covariates/builds/<build_id>/covariates.vrt
```

and a machine-readable manifest:

```text
data/ebird/covariates/builds/<build_id>/manifest.json
```

The VRT presents all bands through one raster interface. Its physical bands
remain tiled, source-specific Cloud-Optimized GeoTIFFs so they can be downloaded,
derived, validated, and replaced independently.

For a regional modeling target, the VRT can be materialized to one layered
BigTIFF/COG:

```text
data/ebird/covariates/builds/<build_id>/exports/
  nc_covariates_2020_2023_250m.tif
```

This satisfies the single-raster modeling interface without making a
multi-decade CONUS file an operational single point of failure.

### Canonical modeling grid

- CRS: CONUS Albers Equal Area, `EPSG:5070`
- Default cell size: 250 m
- Grid alignment: pixel edges snapped to multiples of 250 m from the CRS origin
- Tile size: 100 km by 100 km, or 400 by 400 output cells
- Output data type: `float32`
- Nodata: `-9999.0`
- Internal blocks: 512 by 512 where the tile dimensions permit
- Compression: DEFLATE or ZSTD after compatibility testing
- Overviews: 2, 4, 8, and 16

The 250 m grid balances CONUS size with local habitat representation. Native
30 m categorical products will be aggregated as fractions rather than assigned
by a single nearest pixel. A 100 m NC-only sensitivity can be added if the
250 m representation demonstrably loses barrier-island or narrow-riparian
features.

### Neighborhood scales

Initial derived summaries:

- local: 250 m
- landscape: 1 km
- broader context: 5 km

Categorical products become class fractions. Continuous products use
source-appropriate combinations of mean, standard deviation, minimum, maximum,
or quantiles. Distance features are calculated in `EPSG:5070` and remain in
meters.

## Temporal Contract

Every temporal band has:

- `source_year` or `source_date`
- `valid_from` and `valid_to`
- `cadence`
- `selection_rule`
- `source_age` relative to the modeled observation when extracted

Default temporal selection is the most recent source observation not later than
the checklist date. Future source data are not used unless an explicit
sensitivity enables them.

The raster may contain many temporal bands, but the model-ready extractor
selects only the bands valid for each checklist or locality-season date.
Therefore, a 2021 checklist does not receive every 2020-2023 land-cover or
climate band as simultaneous predictors.

Cadence classes:

- `static`: stable physical or classification layer
- `snapshot`: one mapped state with a source/acquisition date
- `annual`: one valid product per year
- `periodic`: source-specific releases such as five-year land cover
- `monthly`: summaries derived from daily data
- `seasonal`: summaries derived from monthly/daily data using a named scheme

Monthly climate is the canonical portable seasonal representation. Standard
meteorological seasons may also be written. NC biological-season windows remain
a model-side derived view because their boundaries should not silently become a
national ecological assumption.

## Source Inventory and Roles

### Ecological availability channel

| Source | Native cadence/resolution | Planned derived bands |
|---|---|---|
| [USGS Annual NLCD](https://www.usgs.gov/centers/eros/science/annual-nlcd-data-access) | Collection 1.2; annual, 30 m, CONUS, 1985-2025 | Class fractions at 250 m/1 km/5 km; impervious mean; class diversity; fragmentation; annual change derived from adjacent land-cover years; confidence only as optional QA |
| [LANDFIRE vegetation](https://landfire.gov/vegetation) | Release/version based, 30 m | Existing Vegetation Type fractions; tree/shrub/herb cover; vegetation height; disturbance/change metadata where supported |
| [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) | Acquisition mosaic; 1-10 m products | Elevation mean/std; slope; aspect sine/cosine; topographic position; relief; roughness |
| [USGS 3DHP](https://www.usgs.gov/3d-hydrography-program/access-3dhp-data-products) | Annual downloadable snapshots during transition | Stream/river/lake distances; water fractions; drainage density; stream order; catchment context |
| Legacy NHDPlus HR | Frozen fallback where needed | Same hydrography schema with source-family and vintage flags |
| [Daymet](https://daymet.ornl.gov/) | Daily, 1 km, 1980 through latest complete year | Monthly and seasonal temperature, precipitation, vapor pressure, radiation, snow, and day length; 1991-2020 normals; yearly anomalies |
| [USFWS NWI](https://www.fws.gov/program/national-wetlands-inventory/wetlands-data) | Continuously updated distribution; project-specific mapping dates | Cowardin system/subsystem fractions; estuarine/marine/palustrine/lacustrine/riverine fractions; tidal flags; nearest wetland/deepwater distance; mapping-vintage bands |

### Coastal ecological module

| Source | Native cadence/resolution | Planned derived bands |
|---|---|---|
| [NOAA C-CAP regional land cover](https://www.coast.noaa.gov/digitalcoast/data/ccapregional.html) | Approximately five-year cadence, 30 m | Coastal class fractions; estuarine and palustrine wetland fractions; open-water, barren/beach, scrub, forest, and developed fractions; source age |
| [NOAA CUSP](https://nsde.ngs.noaa.gov/) | Continually updated shoreline snapshot | Ocean-facing shoreline distance; shoreline density; contemporary shoreline vintage |
| NWI + C-CAP + CUSP + hydrography | Derived | Back-barrier/sound distance; tidal-wetland fraction; surrounding open-water fraction; local land-strip width; ocean-versus-estuary context |

Barrier-island context will be derived from general land/water topology. No
state-specific island-name or hand-drawn barrier-island flag is required.

### Observer-access channel

| Source | Native cadence | Planned derived bands |
|---|---|---|
| [Census ACS](https://www.census.gov/programs-surveys/acs/data/data-via-api.html) | Annual five-year estimates | Population density, housing density, urban context, source period |
| [Census TIGER/Line](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) | Annual | Road density by class; distance to major and any road; urban-area context |
| [USGS PAD-US](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-download) | Versioned snapshot | Public-access fraction; protected/open-space fraction; distance to public land; owner/manager class |
| Existing eBird sampling summaries | Yearly and historical | Checklist density, observer diversity, locality density, and prior use; retained outside the ecological raster when outcome leakage is possible |

PAD-US and urban/access variables can correlate with habitat, but they remain
observation-process predictors by default. Any availability-side use requires
an explicit ablation and transfer justification.

### Evaluation-only layers

| Source | Use |
|---|---|
| [EPA ecoregions](https://www.epa.gov/eco-research/ecoregions) | Directional transfer splits, summaries, and diagnostics |
| State boundaries, HUCs, Census regions | Tiling, reporting, and held-out-region definitions |

Evaluation-only bands may be rasterized into the logical stack for convenient
joining, but are excluded from model predictors by default.

## Climate Aggregation

Daymet daily variables will be summarized as follows:

| Variable | Monthly summary | Seasonal/yearly summary |
|---|---|---|
| `tmin`, `tmax` | mean, min/max extreme | mean; growing-degree and freeze-day counts |
| precipitation | sum; wet-day count; maximum daily precipitation | sum; anomaly from 1991-2020 normal |
| vapor pressure | mean | mean and anomaly |
| shortwave radiation | mean | mean and anomaly |
| snow-water equivalent | mean and maximum | snow-day count and maximum |
| day length | mean | seasonal mean |

Monthly bands retain year and month. Seasonal bands retain year and named
season. Climate normals and anomalies are separate bands so the model can
distinguish expected regional climate from the conditions in a particular
year.

## Band Naming and Metadata

Stable band identifier:

```text
<channel>__<source>__<variable>__<stat>__r<radius_m>__<time>
```

Examples:

```text
availability__nlcd__deciduous_forest_fraction__mean__r1000__y2021
availability__daymet__precipitation__sum__r1000__y2021m05
availability__nwi__estuarine_fraction__mean__r5000__snapshot
access__tiger__major_road_distance_m__value__r250__y2021
evaluation__epa__ecoregion_level3_code__value__r250__static
```

Each band manifest entry includes:

- band index and identifier
- source and official source URL
- source version/checksum
- channel
- variable/statistic/unit
- source and output resolution
- source date and validity range
- neighborhood radius
- categorical/continuous type
- resampling or aggregation method
- nodata and valid range
- coverage fraction
- derivation code version

The machine-readable manifest contract is checked in at
`scripts/data/ebird_covariates/manifest_schema.json`. The source inventory is
checked in separately at
`scripts/data/ebird_covariates/source_registry.json`, so a source release can
be updated without changing the band contract or orchestration entry point.

## Storage Layout

```text
data/ebird/covariates/
  raw/<source>/<version>/
  staging/<grid_id>/<source>/<version>/
  derived/<grid_id>/<source>/<version>/<tile_id>/
  builds/<build_id>/
    build_config.json
    manifest.json
    covariates.vrt
    tiles/
    exports/
    qa/
  cache/
```

Raw inputs are immutable. Staging normalizes CRS and data typing. Derived files
contain model-scale summaries. Builds assemble a selected temporal range and
source set without copying unchanged raw data.

## Pipeline Stages

1. **Plan**
   - read source registry and build configuration
   - resolve requested years, source releases, AOI, tiles, and expected bands
   - detect unavailable years and temporal leakage before downloading
2. **Fetch**
   - download immutable official source artifacts
   - record URL, request parameters, response metadata, file size, and checksum
   - support retries and resume where the source permits
3. **Normalize**
   - clip or subset by tile with an analysis buffer
   - reproject to `EPSG:5070`
   - preserve categorical codes and source nodata
4. **Derive**
   - calculate terrain, distance, fractions, diversity, fragmentation,
     monthly climate, normals, anomalies, and source-age bands
   - use buffered source tiles to avoid edge artifacts
5. **Validate source blocks**
   - check ranges, coverage, category dictionaries, temporal cadence, and tile
     seams before assembly
6. **Assemble**
   - build the band manifest
   - expose one VRT
   - optionally materialize a regional multi-band BigTIFF/COG
7. **Extract**
   - sample date-valid bands for checklists or locality-season groups
   - keep availability/access/evaluation channels separate
8. **Validate model-ready output**
   - compare source and extracted summaries
   - audit missingness and retained checklist/locality counts
   - generate maps and spot checks

## Source Access Strategy

- Annual NLCD: resolve release-pinned products and filenames from ScienceBase
  metadata. The exposed large-file links are request/file-manager pages, not
  unattended public downloads. Reproducible automated acquisition therefore
  needs either authenticated requester-pays AWS access or a separately resolved
  official MRLC download. The adapter must also accept already acquired local
  archives so source normalization is independent of the acquisition backend.
- LANDFIRE: use full-extent downloads for stable versioned archives or the
  LANDFIRE Product Service API for bounded AOIs.
- 3DEP and National Map products: query the TNMAccess API or staged cloud
  downloads; retain product metadata and acquisition year.
- 3DHP: use versioned annual downloads or official web services. Preserve a
  source-family flag when legacy NHDPlus HR supplies a missing region.
- Daymet: use the NetCDF Subset Service for regional gridded data. The
  single-pixel API is appropriate for verification, not hundreds of thousands
  of checklist requests.
- NWI: download state or HUC8 GeoPackage/FileGDB products and project metadata.
  Deduplicate overlap at state boundaries and retain map-project dates.
- C-CAP: use official regional bulk downloads and their classification tables.
- CUSP: acquire the official shoreline snapshot exposed by the National
  Shoreline Data Explorer and record its retrieval date.
- Census/PAD-US/EPA: use versioned official downloads or APIs and cache the raw
  release artifacts.

Exact machine endpoints belong in the source registry, not hard-coded across
multiple scripts.

## Quality Assurance

### Grid and raster checks

- every derived tile has the expected CRS, resolution, transform, bounds,
  dimensions, nodata, and data type
- adjacent tiles share exact grid edges
- materialized exports have unique band descriptions matching the manifest
- VRT and physical-band sampled values agree
- overviews do not alter base-resolution values

### Value checks

- class fractions are in `[0, 1]`
- mutually exhaustive land-cover fractions sum to approximately one where the
  source is valid
- canopy, shrub, and herb cover remain in valid source ranges
- distances are nonnegative and in true EPSG:5070 meters
- slope, aspect transforms, elevation, and climate variables have plausible
  ranges
- precipitation sums and temperature means use the correct calendar periods
- ecoregion and categorical codes occur in their source dictionaries

### Temporal checks

- no observation receives a source release dated after the modeled date unless
  explicitly permitted
- requested source years and delivered source years are recorded separately
- periodic/snapshot products include source-age bands
- NWI update date is not mistaken for habitat-change date
- climate month/season completeness is checked before aggregation
- leap-day handling is explicit and tested

### Coverage and selection checks

- report valid-cell fraction by source, tile, state, ecoregion, and year
- never drop a checklist solely because one optional module is unavailable
- use missingness and module-coverage bands instead
- audit raw-to-final eBird retention by distance to coast, wetland class,
  ecoregion, locality type, and observer-density stratum

### Geographic spot checks

The NC pilot must include:

- Emerald Isle
- Fort Macon
- Ocracoke
- an inland reservoir
- a mountain locality
- a Piedmont urban locality
- a coastal marsh locality

For barrier-island checks, verify ocean distance, sound/estuary distance,
open-water fraction, tidal-wetland fraction, land-strip width, and land-cover
fractions rather than only visually inspecting one coastline distance.

## Model Integration Contract

The enriched raster changes the availability predictor vector `x_g`; it does
not change the repeated-visit likelihood:

```text
logit(psi_jg) = species intercept + ecological/temporal covariates
logit(p_ijg)  = species intercept + checklist effort/timing + season + frailty
```

Observer-access bands enter the detection/access branch or an explicitly
separate access model. Evaluation-only bands define transfer diagnostics.

The first controlled comparison will use:

- the same species list
- the same locality-season construction
- the same strict support requirements
- the same temporal-regime coastal split
- the same portable no-history and shared-history variants
- the same optimizer, penalties, and epoch count

Only the ecological covariate block changes.

## Promotion Criteria

A source block is promoted only when it provides evidence beyond pooled
checklist ranking:

1. improves or preserves fair any-detection calibration
2. improves species macro ranking/calibration, not only micro metrics
3. improves directional transfer in the regime it represents
4. preserves plausible phenology
5. improves held-out environmental-response behavior
6. reduces high-support failure severity
7. has acceptable coverage and source-age behavior
8. repeats in another state or multi-state region

If an optional coastal module helps NC coastal transfer but is unavailable
inland, it remains a declared module rather than silently changing the national
core.

## Implementation Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Architecture, output contract, source inventory, and ledger | Complete |
| 1 | Source registry, build configuration, fixed-grid planner, and manifest schema | Complete |
| 2 | Windowed/tiled reprojection and VRT/COG assembly | Complete |
| 3 | Annual NLCD and LANDFIRE adapters/derivations | In progress: NLCD catalog complete |
| 4 | 3DEP terrain and 3DHP/NHDPlus hydrography | Pending |
| 5 | NWI, C-CAP, CUSP, and coastal topology | Pending |
| 6 | Daymet monthly/seasonal/normals/anomalies | Pending |
| 7 | Census, TIGER, PAD-US, and evaluation layers | Pending |
| 8 | NC 2020-2023 build, materialized raster, and QA report | Pending |
| 9 | eBird date-aware extraction and locality-season rebuild | Pending |
| 10 | Controlled latent-model reruns and source-block ablations | Pending |
| 11 | Second-region portability test | Pending |
| 12 | CONUS production build | Pending |

## Initial Build Profile

Proposed first reproducible target:

```text
build_id: nc_2020_2023_covariates_v1
aoi: data/boundaries/nc_state_boundary.gpkg
crs: EPSG:5070
resolution_m: 250
years: 2020-2023
neighborhoods_m: 250,1000,5000
logical_output: covariates.vrt
materialized_output: nc_covariates_2020_2023_250m.tif
```

Source blocks should be added and validated in this order:

1. grid, boundary, manifest, and ecoregions
2. Annual NLCD
3. 3DEP terrain
4. 3DHP/NHDPlus hydrography
5. NWI
6. C-CAP and CUSP coastal context
7. LANDFIRE vegetation
8. Daymet temporal climate
9. access/evaluation layers

This order yields an early coastal sensitivity without postponing the more
general CONUS core.

## Planned Command Interface

The implementation should converge on one orchestration entry point:

```text
python scripts/data/ebird-covariates.py plan \
  --config config/ebird_covariates/nc_2020_2023_v1.json

python scripts/data/ebird-covariates.py fetch \
  --config config/ebird_covariates/nc_2020_2023_v1.json

python scripts/data/ebird-covariates.py build \
  --config config/ebird_covariates/nc_2020_2023_v1.json

python scripts/data/ebird-covariates.py validate \
  --build-dir data/ebird/covariates/builds/nc_2020_2023_covariates_v1

python scripts/data/ebird-covariates.py export \
  --build-dir data/ebird/covariates/builds/nc_2020_2023_covariates_v1 \
  --output data/ebird/covariates/builds/nc_2020_2023_covariates_v1/exports/nc_covariates_2020_2023_250m.tif
```

The `plan` command is now runnable. `fetch`, `build`, `validate`, and `export`
remain target interfaces until their corresponding phases are implemented.

The generic Phase 2 raster commands are also runnable. Source adapters will
normally call these functions directly, but the CLI forms are useful for
inspection and one-off source validation:

```text
python scripts/data/ebird-covariates.py normalize-raster \
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json \
  --input <source-raster.tif> \
  --band-id <stable-band-id> \
  --resampling average \
  --output-dir <derived-band-tile-directory>

python scripts/data/ebird-covariates.py assemble-vrt \
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json \
  --band-inventories <band-inventory-1.json> <band-inventory-2.json> \
  --output <covariates.vrt>
```

## Phase 1 Build-Plan Result

The first executable plan completed successfully on 2026-07-17:

```text
python scripts/data/ebird-covariates.py plan \
  --config config/ebird_covariates/nc_2020_2023_v1.json
```

Output:

```text
build_id: nc_2020_2023_covariates_v1
crs: EPSG:5070
resolution: 250 m
snapped bounds: 1054000, 1341750, 1839000, 1690250 m
bounding grid: 3,140 x 1,394 cells (4,377,160 cells)
in-boundary cells (cell-center rule): 2,230,247
fixed 100 km tiles intersecting NC: 27
registered source blocks: 17
estimated logical bands: 1,917
```

The 1,917-band estimate is intentionally conservative and provisional. Daymet
accounts for 1,173 bands because the current registry includes monthly values,
monthly normals/anomalies, and seasonal values/anomalies for 2020-2023. The
registry estimate may change when source releases and class crosswalks are
resolved, but it establishes the correct order of magnitude.

An uncompressed `float32` array would require about 15.9 GiB over cells inside
NC or 31.3 GiB over the rectangular grid. This validates the logical-VRT
decision. Phase 2 will add named materialization profiles so the complete VRT
can expose every band while a model run exports only its declared channels and
temporal views. The plan itself is written to the ignored generated-data path:

```text
data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json
```

The planner currently verifies:

- projected meter-based CRS and exact pixel/tile divisibility
- fixed-origin snapped bounds
- multipart AOI geometry and intersecting tile cells, including coastal islands
- temporal years, months, seasons, and no-future-data selection rule
- unique registered sources and channel assignments
- provisional band expansion across time and neighborhood scales
- source/config checksums and deterministic output locations

No source data are downloaded by `plan`; every adapter is still explicitly
reported by status. Annual NLCD is now `catalog_ready`; the other adapters
remain `planned`.

## Phase 2 Raster-Engine Result

Phase 2 added `scripts/data/ebird_covariates/raster_engine.py` and two CLI
operations:

- `normalize-raster` warps a source band directly onto each fixed tile, masks
  it to the multipart AOI, writes a DEFLATE-compressed COG, and records a band
  inventory.
- `assemble-vrt` mosaics any ordered collection of band inventories into one
  multi-band VRT using exact source/destination windows at edge tiles.

The engine was validated with a synthetic `EPSG:5070` gradient spanning four
independently written tiles. Reopening the VRT produced the same 4 by 4 array as
a direct whole-raster average reprojection, including tile boundaries. The
generated tiles reported `LAYOUT=COG`. The focused validation command is:

```text
python -m unittest tests.test_ebird_covariate_planner \
  tests.test_ebird_covariate_raster_engine \
  tests.test_ebird_covariate_nlcd -v
```

Result: eight tests passed across the planner, raster engine, and Annual NLCD
catalog adapter. These tests cover grid snapping, temporal expansion,
registry/config validation, COG creation, VRT assembly, CRS/dimensions, band
descriptions, value equivalence across seams, release-pinned NLCD filenames,
product/year selection, and acquisition-status metadata.

## Phase 3 Annual NLCD Catalog Result

The first production source adapter pins Annual NLCD Collection 1.2 and
resolves exact official ScienceBase file metadata without downloading raster
archives. The NC build configuration selects only the source products needed
for the first controlled model comparison:

- `LndCov`, 2019-2023; 2019 is retained only as the predecessor needed to derive
  2020 land-cover change without using future information.
- `FctImp`, 2020-2023.

The metadata-only command is:

```text
python scripts/data/ebird-covariates.py catalog-nlcd \
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json
```

Resolved result:

```text
release: C1V2
files: 9
LndCov: 5 files, 6.6 GiB
FctImp: 4 files, 3.6 GiB
total source archives: 10.2 GiB
acquisition status: metadata_resolved_manual_or_aws_credentials_required
```

An exploratory five-product catalog (`LndCov`, `LndCnf`, `FctImp`, `ImpDsc`,
and `LndChg`) would have required 45.9 GiB. It was not promoted. Confidence is
an optional QA product, impervious descriptors are deferred, and annual change
will be derived from adjacent `LndCov` years. This is a storage and provenance
decision, not a claim that the deferred products have no scientific value.

The catalog schema deliberately records ScienceBase file-manager,
download-request, and download-URI fields separately and sets
`direct_download_available=false`. It does not call any of them a direct
download URL. Acquired archives must receive a local SHA-256 because these
large-file metadata records do not provide remote checksums.

## Decision Log

### 2026-07-17

- Use `EPSG:5070`, not EPSG:3857, for all model-scale distances and areas.
- Use a 250 m canonical grid with 250 m, 1 km, and 5 km summaries.
- Preserve native-resolution source archives.
- Provide one logical VRT plus a manifest; materialize a single layered raster
  for regional builds.
- Store monthly climate as the portable seasonal primitive and derive named
  seasons explicitly.
- Use the latest source not later than an observation date by default.
- Keep access/observer geography separate from ecological availability.
- Treat NWI mapping vintage and C-CAP cadence as provenance, not annual change.
- Pilot in NC without overwriting the coarse four-covariate dataset.
- Require a second-region replication before CONUS production.
- Register all 17 initial source/derived blocks in one versioned registry.
- Keep the all-band product as a VRT and use named regional export profiles;
  the NC plan showed that blind materialization would be unnecessarily large.
- Normalize source bands as independent fixed-grid COG tiles and assemble the
  logical stack from band inventories; the synthetic seam test showed exact
  agreement with whole-raster reprojection.
- Pin Annual NLCD Collection 1.2 and use `LndCov` plus `FctImp` for the first
  pilot rather than automatically acquiring every available product.
- Derive annual land-cover change from adjacent land-cover years; acquire 2019
  `LndCov` as the predecessor for the 2020 model year.
- Keep metadata resolution separate from acquisition. ScienceBase currently
  supplies authoritative file metadata but not an unattended public large-file
  path for this adapter.

## Open Questions

1. Whether the 250 m grid needs a 100 m coastal sensitivity.
2. Whether authenticated requester-pays AWS or official MRLC downloads should
   be the primary Annual NLCD acquisition backend; local archive ingestion will
   be supported in either case.
3. Whether LANDFIRE EVT should be retained at detailed class level or collapsed
   to a portable ecological hierarchy.
4. How to crosswalk 3DHP and legacy NHDPlus HR attributes into one schema.
5. Which NWI Cowardin hierarchy best balances ecological meaning and sparsity.
6. Whether the NC materialized output should include all monthly climate bands
   or whether they should remain in the logical VRT and date-aware extractor.
7. Which second region provides the strongest portability test without
   requiring a full CONUS eBird build.

## Next Ledger Update

The next update should record:

- the selected Annual NLCD acquisition backend and immutable local archive
  checksums
- the first real NC NLCD source and derived tiles
- exact class-fraction, diversity, fragmentation, imperviousness, and adjacent-
  year change derivations
- named materialization profiles and exact inclusion rules
- coverage, value-range, class-fraction-sum, and geographic spot-check results
- source-window/subset behavior and measured disk/runtime costs
