# eBird CONUS Covariate Pipeline

## Status

- Started: 2026-07-17
- Current phase: Phase 3 in progress; Annual NLCD is promoted after its
  corrected `all_touched` full build passed code, numerical, mapped, and
  statewide checklist-support gates. LANDFIRE cataloging, release-specific
  class tables, portable crosswalks, and bounded interior/coastal raw and
  model-scale LF2023 pilots are validated; annual disturbance and multi-release
  state scaling are next
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
| 3 | Annual NLCD and LANDFIRE adapters/derivations | In progress: Annual NLCD promoted for NC; LANDFIRE metadata, crosswalk, raw export, model-scale derivation, and QA pass on bounded interior/coastal LF2023 pilots; annual disturbance and state scaling remain |
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

Original output under the cell-center AOI rule:

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

Statewide checklist QA later showed that the cell-center rule can mask a 250 m
pixel even when a valid checklist point lies inside a thin boundary sliver of
the AOI. The checked-in regional config now declares
`grid.aoi_mask_rule = "all_touched"`. A new plan records center-rule,
all-touched-rule, and selected-rule cell counts separately. Replanning retained
the same 27 tiles and selected 2,235,542 all-touched AOI cells, 5,295 more than
the original 2,230,247 center-rule cells (`+0.2374%`). This is a narrow
perimeter correction, not a material study-area expansion. Exact point/AOI
membership remains a vector polygon operation and is not inferred from the
raster mask.

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
- an explicit `center` or `all_touched` AOI raster-mask contract, with both
  active-cell counts retained for auditability
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

- `normalize-raster` warps a source band directly onto each fixed tile, applies
  the plan's explicit AOI mask rule, writes a DEFLATE-compressed COG, and
  records a band inventory.
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

## Phase 3 Annual NLCD Derivation Checkpoint: 2026-07-20

The production catalog command was rerun successfully against the NC plan. It
resolved the intended nine Collection 1.2 archives and wrote
`data/ebird/covariates/raw/annual_nlcd/C1V2/catalog.json`:

```text
Annual NLCD C1V2: 9 files, 10.2 GiB total
LndCov: 5 files, 6.6 GiB
FctImp: 4 files, 3.6 GiB
acquisition status: metadata_resolved_manual_or_aws_credentials_required
```

The acquisition status is expected. Cataloging resolves authoritative metadata
and deterministic raster identities; it does not claim that the large raster
objects are locally present or that requester-pays AWS credentials work.

The metadata-only AWS registration was also run successfully and wrote
`data/ebird/covariates/raw/annual_nlcd/C1V2/sources.aws.json` with nine
requester-pays references in `us-west-2`. The first and last registered raster
URIs correspond to 2019 `LndCov` and 2023 `FctImp`.

Requester-pays authentication and official-source header validation then
passed on 2026-07-20. All nine remote rasters opened successfully, all five
annual land-cover rasters use the same 30 m grid, and the validation artifact
was written to
`data/ebird/covariates/raw/annual_nlcd/C1V2/sources.aws.validation.json`:

```text
Validated 9 Annual NLCD rasters; land-cover grids aligned=True
```

This closes the source-access gate. It validates identities and raster grid
metadata, but it does not yet validate derived NC values, window-read cost,
runtime, tile continuity, or coastal/island behavior.

Implemented source operations:

- `register-nlcd-local` finds either extracted TIFFs or official ZIP archives,
  verifies the exact TIFF member, opens it through Rasterio/GDAL, records grid
  metadata, checks catalog archive size where applicable, and optionally
  calculates SHA-256.
- `register-nlcd-aws` writes requester-pays `/vsis3/` COG references without
  falsely marking them as opened or validated.
- `validate-nlcd-sources` opens every registered raster, verifies one-band
  metadata, and checks that all annual land-cover grids align. This is the
  required lightweight gate before derivation.
- `derive-nlcd` performs buffered source-window reads, area-preserving average
  reprojection, fractional cell-overlap circular-neighborhood aggregation,
  plan-controlled AOI masking, tiled COG inventory writes, completeness
  validation, and `annual_nlcd.vrt` assembly.
  New runs also persist COG count/payload and elapsed time in the derivation
  summary and print them to the console.
- `validate-nlcd-derived` verifies the derived COG grid/data contract, value
  ranges, inventory completeness, and per-year/radius class-fraction sums. It
  writes a machine-readable validation artifact and optional four-panel mapped
  previews for selected plan tiles.

The derivation definitions are now explicit:

- class fraction: valid source-pixel area in each of the 16 modeled Annual NLCD
  classes divided by valid modeled-class area within the circular neighborhood
- diversity: Shannon entropy across the 16 class fractions, normalized by
  `log(16)`
- fragmentation/mixing index: `1 - max(class fraction)`; this is not a patch-
  edge or connected-component fragmentation metric
- annual change: area whose aligned land-cover class differs between `year - 1`
  and `year`, divided by area valid in both years
- imperviousness: native 0-100 fractional imperviousness converted to 0-1 and
  averaged by valid source area
- source coverage: valid modeled land-cover area at the smallest configured
  neighborhood, retained as a QA/missingness band

For 2020-2023 and 250 m, 1 km, and 5 km neighborhoods, the exact band count is:

```text
16 class fractions x 3 radii x 4 years = 192
2 diversity/fragmentation x 3 radii x 4 years = 24
source coverage x 4 years = 4
annual change x 3 radii x 4 years = 12
imperviousness x 3 radii x 4 years = 12
total = 244 bands
```

Synthetic validation uses aligned 2019/2020 land-cover rasters and a 2020
impervious raster split across four independently derived output tiles. It
verifies reprojection, fractional circular aggregation, class fractions
summing to one, directional change/impervious contrasts, buffered seam
continuity, COG inventories, exact expected band count, and a readable VRT.
The complete covariate test suite now passes:

```text
env\Scripts\python.exe -m unittest \
  tests.test_ebird_covariate_planner \
  tests.test_ebird_covariate_raster_engine \
  tests.test_ebird_covariate_nlcd \
  tests.test_ebird_covariate_nlcd_derive -v
```

Result: 11 tests passed. This validates code behavior on controlled rasters; it
does not substitute for validating the official raster objects or inspecting
the NC outputs.

### Official source access gate

Requester-pays AWS is the preferred scalable backend if standard AWS
credentials are available. Registration itself is metadata-only:

```text
env\Scripts\python.exe scripts/data/ebird-covariates.py register-nlcd-aws \
  --catalog data/ebird/covariates/raw/annual_nlcd/C1V2/catalog.json \
  --output data/ebird/covariates/raw/annual_nlcd/C1V2/sources.aws.json
```

Then open only the nine raster headers before any heavy processing:

```text
env\Scripts\python.exe scripts/data/ebird-covariates.py validate-nlcd-sources \
  --sources data/ebird/covariates/raw/annual_nlcd/C1V2/sources.aws.json
```

If requester-pays access is unavailable, download the nine cataloged official
ZIPs or TIFFs into one immutable local directory, then use:

```text
env\Scripts\python.exe scripts/data/ebird-covariates.py register-nlcd-local \
  --catalog data/ebird/covariates/raw/annual_nlcd/C1V2/catalog.json \
  --input-dir <annual-nlcd-download-directory> \
  --sha256 \
  --output data/ebird/covariates/raw/annual_nlcd/C1V2/sources.local.json

env\Scripts\python.exe scripts/data/ebird-covariates.py validate-nlcd-sources \
  --sources data/ebird/covariates/raw/annual_nlcd/C1V2/sources.local.json
```

After validation passes, run one complete 100 km plan tile before the full
27-tile NC derivation. The pilot retains all four years, all three neighborhood
radii, and the exact 244-band contract while bounding the first remote-window
read. `xp0014_yp0015` is a fully active interior tile:

```text
set AWS_PROFILE=ebird-nlcd
set AWS_DEFAULT_REGION=us-west-2
env\Scripts\python.exe scripts/data/ebird-covariates.py derive-nlcd ^
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json ^
  --sources data/ebird/covariates/raw/annual_nlcd/C1V2/sources.aws.json ^
  --tile-ids xp0014_yp0015 ^
  --output-dir data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/annual_nlcd
```

The official-source interior pilot completed successfully. It produced the
expected 244 bands and 244 derived COGs for `xp0014_yp0015`. The COG payload is
151.28 MiB, excluding the small inventories, VRT, and diagnostics. Reusable QA
reported:

```text
Validated derived Annual NLCD: 244 bands, 244 COGs;
maximum class-fraction sum error=7.15e-07
All checks passed: True
```

All derived values were within `[0, 1]` under the declared `1e-5` floating-
point tolerance, the fully active interior tile had source coverage 1.0, and
all files matched the EPSG:5070 400 x 400 tiled-float32 contract. The 2023
preview showed coherent land-cover mosaics, concentrated urban imperviousness,
mostly low annual change, and nonblank aligned panels. Runtime was not captured
for this first command and must be measured on a later bounded or full run.

The exact reusable QA command is:

```text
env\Scripts\python.exe scripts/data/ebird-covariates.py validate-nlcd-derived ^
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json ^
  --summary data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/annual_nlcd/annual_nlcd_summary.json ^
  --preview-tile-ids xp0014_yp0015
```

This passes the interior numerical gate but does not test North Carolina's
barrier-island geometry. Before the full state, rebuild a coherent two-tile
pilot containing the completed interior tile and `xp0017_yp0014`, the 100 km
tile selected for Emerald Isle/Fort Macon and adjacent sound/ocean context. It
contains 48,083 active 250 m AOI cells. The derivation summary and inventories
describe the selected tile set, so include both IDs rather than running the
coastal tile alone:

```text
set AWS_PROFILE=ebird-nlcd
set AWS_DEFAULT_REGION=us-west-2
env\Scripts\python.exe scripts/data/ebird-covariates.py derive-nlcd ^
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json ^
  --sources data/ebird/covariates/raw/annual_nlcd/C1V2/sources.aws.json ^
  --tile-ids xp0014_yp0015 xp0017_yp0014 ^
  --output-dir data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/annual_nlcd

env\Scripts\python.exe scripts/data/ebird-covariates.py validate-nlcd-derived ^
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json ^
  --summary data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/annual_nlcd/annual_nlcd_summary.json ^
  --preview-tile-ids xp0014_yp0015 xp0017_yp0014
```

The two-tile coastal gate passed:

```text
Derived Annual NLCD C1V2: 244 bands across 2 plan tiles
Validated derived Annual NLCD: 244 bands, 488 COGs;
maximum class-fraction sum error=7.15e-07
All checks passed: True
Minimum AOI support xp0017_yp0014:
  r250=99.39%, r1000=98.01%, r5000=90.42%
Minimum AOI support xp0014_yp0015:
  r250=100.00%, r1000=100.00%, r5000=100.00%
```

The combined COG payload is 196.36 MiB, so the coastal tile added only about
45.08 MiB. Its preview preserves visible barrier-island, sound, shoreline,
developed-area, and open-water structure without misalignment or leakage
outside the NC AOI. Larger-radius support decreases over state-water/ocean
cells as expected, rather than because of a raster seam or reprojection error.

Checklist-location QA provides the more relevant modeling gate. The coastal
tile contains 15,228 processed checklists at 4,159 localities. Annual NLCD
class-fraction support is available for 15,227/15,228 checklists at 250 m and
1 km and 15,225/15,228 at 5 km. All 2,837 checklists whose locality labels
match Emerald Isle or Fort Macon have complete 250 m source coverage. The
three unsupported records are private traveling checklists with reported
routes of 1.239-5.633 km and plotted locations 3.49-6.66 km seaward of the
processed coastline metric. Two locality names explicitly identify the
Atlantic Ocean; the third is a one-use Morehead personal location. Treat the
first two as likely marine checklist events and the third as marine-likely but
less certain. This is a checklist-level classification, not evidence that
every detected bird occupied the single plotted coordinate. eBird instructs
users to plot a traveling checklist near the route midpoint when a mobile GPS
track is unavailable, while the track is the more precise representation of
where the observer traveled ([eBird best practices](https://support.ebird.org/en/support/solutions/articles/48000795623-ebird-rules-and-best-practices),
[mobile tracks](https://support.ebird.org/en/support/solutions/articles/48000960508-ebird-mobile-tips-tricks)). Preserve Annual NLCD values there as missing
*terrestrial* ecological support rather than converting them to land-cover
zeroes. This result does not justify a statewide 100 m sensitivity build;
retain that option for a later failure at Ocracoke or another narrow-island
transfer test.

The July 20 Cost Explorer export also showed two new US West Tier-2 request
lines totaling about `$0.0000776`, with no billed data transfer. Assuming no
other US West S3 reads that day, this is the likely combined source-validation
and one-tile pilot request cost. The estimated full-NC request cost is about
`$0.00182-$0.00210`, with `$0.00233` as a conservative 30x budget. This is not
a material cost gate, but the full run should still record its posted charge.

Do not add `--overwrite`. Existing pilot COGs will not be overwritten, although
the derivation currently rereads/recomputes source windows while rebuilding
complete inventories. The coastal gate authorizes the same output directory
with `--tile-ids` omitted for the full build:

```text
env\Scripts\python.exe scripts/data/ebird-covariates.py derive-nlcd ^
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json ^
  --sources data/ebird/covariates/raw/annual_nlcd/C1V2/sources.aws.json ^
  --output-dir data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/annual_nlcd
```

The full command completed successfully:

```text
Derived Annual NLCD C1V2: 244 bands across 27 plan tiles
COG payload: 6,588 files, 2108.33 MiB
Elapsed time: 1993.3 seconds
Validated derived Annual NLCD: 244 bands, 6588 COGs;
maximum class-fraction sum error=9.54e-07
All checks passed: True
```

This is a 2.06 GiB derived COG payload and a 33 minute 13 second full-state
run. Every inventory, COG grid contract, value-range check, and class-fraction
sum passed. Interior tiles generally have 100% support at all three radii. The
lowest state-edge support occurs on ocean-heavy `xp0018_yp0016`: 98.86% at
250 m, 96.50% at 1 km, and 83.50% at 5 km. That radius-dependent decline is
consistent with terrestrial-source support ending offshore, not a processing
failure.

Five generated previews were inspected: `xp0014_yp0015`, `xp0017_yp0014`,
`xp0017_yp0015`, `xp0017_yp0016`, and `xp0018_yp0015`. They show coherent
interior land-cover and impervious patterns, visible estuarine and tidal-wetland
structure, retained barrier-island geometry, and AOI masking that follows the
state boundary without an apparent tile seam. The 250 m grid therefore passes
the current full-state visual gate; a 100 m product remains a targeted later
sensitivity rather than a prerequisite.

The statewide support gate was run at all processed checklist coordinates:

```text
env\Scripts\python.exe scripts/data/ebird-covariates.py validate-nlcd-checklist-support ^
  --summary data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/annual_nlcd/annual_nlcd_summary.json ^
  --checklists data/ebird/processed_nc_2020_2023/checklists.geoparquet
```

It writes `annual_nlcd_checklist_support.json` and
`annual_nlcd_unsupported_checklists.csv` below the Annual NLCD diagnostics
directory. The JSON reports support by year and radius; the CSV preserves the
event, locality, protocol, effort distance, and coastline-distance context for
every unsupported event. This lets marine-likely records remain declared
terrestrial missingness while exposing any unexpected unsupported terrestrial
or barrier-island checklist.

The first full-state result was:

```text
Annual NLCD checklist support: 661,979/661,979 checklists eligible
  r250: 661,815/661,979 supported (99.9752%); 164 unsupported
  r1000: 661,815/661,979 supported (99.9752%); 164 unsupported
  r5000: 661,809/661,979 supported (99.9743%); 170 unsupported
```

The percentage alone is not a sufficient promotion criterion because the
failures are geographically structured. Inspection separated two mechanisms:

- 164 checklists at 65 localities lacked all three radii. Every coordinate is
  covered by the exact NC AOI polygon, but each lies only 4.39-98.11 m inside
  its boundary (median 13.76 m). These include mountain and state-border
  localities; 70 records are from Great Smoky Mountains NP--Charlies Bunion
  Trail. They are not missing Annual NLCD source data. Their containing 250 m
  raster-cell centers fall outside the AOI and were masked by the center rule.
- Six checklists at four explicitly marine/pelagic localities retain 250 m and
  1 km support but lack 5 km support. These include Hatteras Pelagic and
  Atlantic/North Atlantic Ocean locations. That is expected terrestrial-source
  support behavior and should remain missing rather than be imputed as zero.

Therefore the current center-masked raster is a **provisional numerical pass,
not a promoted model input**. The correction is a grid-contract change, not
nearest-cell imputation: retain every target pixel touched by the regional AOI,
while continuing to use the exact vector polygon for point membership and
reporting. This is appropriate for a reusable regional 250 m covariate stack
because boundary pixels represent partial AOI support and may also be needed
for neighborhood summaries. It does not assert that the entire pixel belongs
to NC.

The planner, generic raster engine, Annual NLCD derivation, and Annual NLCD QA
now share the declared `all_touched` rule. Plans retain both center and
all-touched active-cell counts. New COGs and summaries record the rule, and
reuse of a COG generated under a different rule fails with an explicit
`--overwrite` instruction. Run the focused regression suite before rebuilding:

```text
env\Scripts\python.exe -m unittest ^
  tests.test_ebird_covariate_planner ^
  tests.test_ebird_covariate_raster_engine ^
  tests.test_ebird_covariate_nlcd ^
  tests.test_ebird_covariate_nlcd_derive -v
```

Result: all 14 tests passed in 2.744 seconds. This covers all-touched planning
and masking, stale mask-contract rejection in both raster paths, tiled COG/VRT
assembly, Annual NLCD catalog/registration behavior, fractional neighborhood
aggregation, and checklist-support eligibility/masking behavior. The code gate
passes; the generated NC plan and COGs still use the old contract until the
commands below are run.

Then regenerate the plan so it contains the new mask contract and counts:

```text
env\Scripts\python.exe scripts/data/ebird-covariates.py plan ^
  --config config/ebird_covariates/nc_2020_2023_v1.json ^
  --output data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json
```

The revised plan passed with unchanged snapped bounds, bounding dimensions,
source/band contract, and 27-tile layout:

```text
AOI cells (all_touched): 2,235,542; tiles: 27
Sources: 17; estimated logical bands: 1,917
Uncompressed float stack estimate: 16.0 GiB over AOI cells;
31.3 GiB over bounding grid
```

Compared with the historical center-rule plan, only 5,295 cells (`0.2374%`)
were added. This bounded change is consistent with the diagnosed boundary
localities and does not trigger a resolution or tiling redesign.

This contract change is the explicit exception to the normal no-overwrite
rule. Rebuild every Annual NLCD COG so old center-masked and new all-touched
tiles cannot be mixed:

```text
set AWS_PROFILE=ebird-nlcd
set AWS_DEFAULT_REGION=us-west-2
env\Scripts\python.exe scripts/data/ebird-covariates.py derive-nlcd ^
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json ^
  --sources data/ebird/covariates/raw/annual_nlcd/C1V2/sources.aws.json ^
  --output-dir data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/annual_nlcd ^
  --overwrite
```

The corrected all-touched overwrite completed successfully:

```text
Derived Annual NLCD C1V2: 244 bands across 27 plan tiles
COG payload: 6,588 files, 2112.22 MiB
Elapsed time: 2227.7 seconds
```

The persisted summary confirms `aoi_mask_rule = "all_touched"`, all 244
expected bands, all 27 plan tiles, and all 6,588 COGs. Relative to the original
center-mask build, payload increased by only 3.89 MiB (`0.1845%`). Runtime was
37 minutes 7.7 seconds, 234.4 seconds (`11.8%`) longer than the original
33-minute build. Neither change is large enough to alter the grid or processing
strategy. Derivation completion alone was not promotion: numerical, mapped,
and checklist-support QA still had to pass on the rewritten files.

Finally rerun raster and checklist QA:

```text
env\Scripts\python.exe scripts/data/ebird-covariates.py validate-nlcd-derived ^
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json ^
  --summary data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/annual_nlcd/annual_nlcd_summary.json ^
  --preview-tile-ids xp0010_yp0013 xp0010_yp0014 xp0014_yp0015 xp0017_yp0014 xp0017_yp0015 xp0017_yp0016 xp0018_yp0015

env\Scripts\python.exe scripts/data/ebird-covariates.py validate-nlcd-checklist-support ^
  --summary data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/annual_nlcd/annual_nlcd_summary.json ^
  --checklists data/ebird/processed_nc_2020_2023/checklists.geoparquet
```

The corrected numerical and mapped gate passed:

```text
Validated derived Annual NLCD: 244 bands, 6588 COGs;
maximum class-fraction sum error=9.54e-07
All checks passed: True
```

Minimum AOI support remains 100% across most western and interior tiles. As
expected, support declines with neighborhood radius in ocean-heavy eastern
tiles. The lowest reported tile is `xp0018_yp0016`, with 97.73% support at
250 m, 95.40% at 1 km, and 82.54% at 5 km. This is not evidence of a failed
derivation: the seven representative previews show coherent western AOI
slivers, continuous interior land cover, retained barrier islands and
estuarine/open-water structure, and no visible tile seams or fill artifacts.
The radius-dependent eastern decline is therefore retained as explicit
terrestrial-source support truncation over ocean rather than filled with
land-cover zeroes. The corrected raster is now accepted; checklist-location
support remains the final Annual NLCD promotion gate.

The post-rebuild checklist-location gate then passed:

```text
Annual NLCD checklist support: 661,979/661,979 checklists eligible
  r250: 661,978/661,979 supported (99.9998%); 1 unsupported
  r1000: 661,978/661,979 supported (99.9998%); 1 unsupported
  r5000: 661,972/661,979 supported (99.9989%); 7 unsupported
```

This removes the diagnosed mask artifact: all-radius failures fell from 164 to
one. The remaining seven events are all marine or marine-likely traveling
checklists. Six retain 250 m and 1 km support but lose 5 km support at named
Hatteras Pelagic, Atlantic Ocean, or North Atlantic Ocean locations plotted
3.49-5.96 km seaward. The sole all-radius failure is a private Morehead
traveling checklist plotted 6.66 km seaward, outside terrestrial NLCD source
support. By year, 2021-2023 have complete 250 m and 1 km support; the one
all-radius marine-likely record is in 2020. These are declared terrestrial
missingness, not ecological zeroes. The correction therefore resolves every
identified terrestrial/boundary failure, and Annual NLCD is promoted for the
NC enriched-covariate build.

Replace `sources.aws.json` with `sources.local.json` for local inputs. Do not
normally add `--overwrite`; valid tiles can then be reused after interruption.
The one required overwrite above is scoped to the AOI mask-contract migration.
Cost Explorer subsequently posted same-day request charges of approximately
`$0.0011744` for the full-build day (`$0.0011496` internal Tier 2 plus
`$0.0000248` Tier 2), with no data-transfer-out charge. The daily total was
`$0.0012768` after unrelated timed-storage cost. This is account/day-level
billing evidence rather than an instrumented per-command invoice, but it
confirms that requester-pays source reads were operationally negligible for
the NC build.

## Phase 3 LANDFIRE Catalog Checkpoint: 2026-07-22

The release-aware LANDFIRE catalog adapter is implemented in
`scripts/data/ebird_covariates/landfire.py` and exposed as
`catalog-landfire`. It resolves the current official LFPS product inventory,
selects exact CONUS layers, validates every ArcGIS ImageServer, and writes an
immutable metadata snapshot. It does not submit an LFPS job, require an email
address, use AWS, or download raster pixels.

The NC retrospective selection policy is explicit:

- vegetation products: EVT, EVC, and EVH
- vegetation releases: `LF2016`, `LF2022`, and `LF2023`
- observation-year mapping: 2020 -> `LF2016`, 2021 -> `LF2016`, 2022 ->
  `LF2022`, and 2023 -> `LF2023`
- vegetation source ages: 4, 5, 0, and 0 years respectively
- annual disturbance products: `LF2020_Dist20`, `LF2022_Dist21`,
  `LF2022_Dist22`, and `LF2023_Dist23`

LF2020 vegetation is archived and absent from the current LFPS product
inventory. The pilot therefore uses the latest publicly available non-future
vegetation release, LF2016, for 2020-2021 and retains source age rather than
silently relabeling the source as annual. An archived LF2020 acquisition can
replace this fallback later without changing the temporal contract.

The planner now expands release-axis products by the configured release list
and treats annual disturbance as a year-axis product. The NC plan consequently
increases from 1,917 to 2,020 estimated logical bands; LANDFIRE accounts for
153 bands. This estimate is intentionally conservative and precedes named
materialization profiles.

Exact commands and accepted output:

```bat
env\Scripts\python.exe scripts/data/ebird-covariates.py plan ^
  --config config/ebird_covariates/nc_2020_2023_v1.json ^
  --output data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json

env\Scripts\python.exe scripts/data/ebird-covariates.py catalog-landfire ^
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json
```

```text
Resolved LANDFIRE: 13 official layers (9 vegetation, 4 disturbance)
Vegetation release mapping: 2020->LF2016, 2021->LF2016, 2022->LF2022, 2023->LF2023
Validated public ImageServers: EPSG:5070 at 30 m; all passed=True
Acquisition status: official_public_imageservers_resolved
Wrote catalog to data\ebird\covariates\raw\landfire\catalog.json
```

The live LFPS inventory contained 136 records. All 13 selected services are
one-band, 30 m thematic rasters in `EPSG:5070` with nearest-neighbor default
resampling. Annual disturbance services live under the separate official
`Landfire_Disturbance` folder; the adapter resolves that distinction
explicitly. Do not derive fractions from display colors or bilinearly resample
categorical source codes.

## Phase 3 LANDFIRE Attribute and Raster Checkpoint: 2026-07-22

Release-specific EVT, EVC, and EVH class tables are now acquired from the
official ImageServer `rasterAttributeTable` resource using each service's
named raster function. This is preferable to reading VAT members from the
multi-gigabyte full-extent ZIPs: the archive host accepts suffix ranges but
returns HTTP 500 for absolute byte offsets above 2 GiB. The structured service
returns the exact class fields directly and transfers only about 1.5 MiB for
all nine tables.

The accepted table inventory is:

- EVT: 857 rows for LF2016 and 831 rows each for LF2022 and LF2023
- EVC: 264, 266, and 266 rows
- EVH: 103, 104, and 105 rows
- total: 3,627 rows across nine raw JSON and normalized CSV artifacts

The portable EVT hierarchy has nine classes: `forest_tree`, `shrub`,
`herbaceous`, `riparian`, `agriculture`, `developed`,
`sparse_barren`, `open_water`, and `snow_ice`. It is derived from the
official `EVT_PHYS` field. The only disambiguation uses `EVT_LF` to split
`Exotic Tree-Shrub` rows into tree or shrub. Raw release, value, name,
lifeform, physiognomy, and other official fields remain in the crosswalk.
Unknown future categories fail validation instead of being assigned by
keyword fallback.

EVC and EVH require a narrower interpretation. LANDFIRE describes each as a
single composite raster produced from lifeform-specific layers. Therefore the
lookups record cover or height conditional on the mapped dominant lifeform;
they do not claim simultaneous tree, shrub, and herb strata at a pixel. The
crosswalk contains 731 numeric EVC rows and 247 numeric EVH rows, retains
special developed/agricultural/water classes separately, and marks lower-bound
classes such as `Herb Cover >= 99%`.

Exact commands:

```bat
env\Scripts\python.exe scripts/data/ebird-covariates.py catalog-landfire-attributes ^
  --catalog data/ebird/covariates/raw/landfire/catalog.json

env\Scripts\python.exe scripts/data/ebird-covariates.py build-landfire-crosswalks ^
  --attributes-summary data/ebird/covariates/raw/landfire/attributes/landfire_attribute_tables.json

env\Scripts\python.exe scripts/data/ebird-covariates.py export-landfire ^
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json ^
  --catalog data/ebird/covariates/raw/landfire/catalog.json ^
  --crosswalk-summary data/ebird/covariates/raw/landfire/crosswalks/landfire_crosswalk_summary.json ^
  --tile-ids xp0014_yp0015 ^
  --layers LF2023_EVT LF2023_EVC LF2023_EVH ^
  --buffer-m 5000 ^
  --output-dir data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/landfire/raw
```

Accepted bounded-pilot output:

```text
LANDFIRE export tile 1/1: xp0014_yp0015
  xp0014_yp0015 LF2023_EVT: 3668x3668, 74 values, 100.00% coverage, 26.3 MiB
  xp0014_yp0015 LF2023_EVC: 3668x3668, 229 values, 100.00% coverage, 26.3 MiB
  xp0014_yp0015 LF2023_EVH: 3668x3668, 82 values, 100.00% coverage, 26.3 MiB
Exported 3 validated LANDFIRE rasters; all checks passed=True
```

The adapter snaps the requested tile plus 5 km halo to the native LANDFIRE
grid, requests nearest-neighbor one-band TIFFs in `EPSG:5070`, and checks
dimensions, transform, bounds, source coverage, and every observed class code
against the exact release lookup. The three interior-pilot files total
78.86 MiB. The same command then passed for coastal tile
`xp0017_yp0014`: EVT, EVC, and EVH again measured `3668 x 3668`, had
100% source coverage, and contained 55, 209, and 82 registered values
respectively. The two pilots comprise six files and 157.72 MiB. LANDFIRE
therefore covers this ocean-adjacent test cleanly; the remaining Annual NLCD
marine gaps are source-specific rather than evidence of a shared grid or AOI
failure. The next gate is deriving the nine EVT fractions and conditional
EVC/EVH lifeform summaries at 250 m, 1 km, and 5 km on both pilots.

## Phase 3 LANDFIRE Model-Scale Derivation Checkpoint: 2026-07-22

The LF2023 model-scale vegetation derivation and reusable QA command are now
implemented. Each release has a fixed 46-band logical schema:

- 27 EVT bands: nine portable class fractions at 250 m, 1 km, and 5 km
- nine EVC bands: conditional dominant-tree, shrub, and herb cover means at
  the three radii
- nine EVH bands: conditional dominant-tree, shrub, and herb height means at
  the three radii
- one 250 m modeled-source coverage band

Conditional EVC/EVH values require at least 80% total modeled source coverage
and at least 1% support from the named lifeform. Other lifeforms are excluded
from the denominator rather than assigned structural zeroes. A band that is
all NoData on one tile remains in the inventory and VRT schema but does not
produce an unnecessary physical COG. This sparse-tile rule is required for a
stable CONUS schema.

Exact accepted commands for the interior pilot were:

```bat
env\Scripts\python.exe scripts/data/ebird-covariates.py derive-landfire ^
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json ^
  --export-summary data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/landfire/raw/landfire_export_summary.json ^
  --crosswalk-summary data/ebird/covariates/raw/landfire/crosswalks/landfire_crosswalk_summary.json ^
  --output-dir data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/landfire/derived_interior

env\Scripts\python.exe scripts/data/ebird-covariates.py validate-landfire-derived ^
  --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json ^
  --summary data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/landfire/derived_interior/landfire_derived_summary.json ^
  --preview
```

```text
Derived LANDFIRE LF2023 xp0014_yp0015: 46 bands, 44 COGs, 23.23 MiB
Validated derived LANDFIRE LF2023 xp0014_yp0015: 46 bands, 44 COGs,
  2 empty logical bands; maximum EVT fraction-sum error=7.15e-07
  r250: AOI support=100.00%
  r1000: AOI support=100.00%
  r5000: AOI support=100.00%
All checks passed: True
```

The two empty interior bands are the 5 km conditional shrub-cover and
shrub-height bands. Shrub support does not reach the 1% threshold anywhere in
that tile at 5 km, so physical all-NoData files would add no information.

The corresponding coastal derivation used the export summary below
`sources/landfire/raw_coastal` and wrote to
`sources/landfire/derived_coastal`:

```text
Derived LANDFIRE LF2023 xp0017_yp0014: 46 bands, 46 COGs, 6.50 MiB
Validated derived LANDFIRE LF2023 xp0017_yp0014: 46 bands, 46 COGs,
  0 empty logical bands; maximum EVT fraction-sum error=7.15e-07
  r250: AOI support=99.04%
  r1000: AOI support=97.81%
  r5000: AOI support=90.43%
All checks passed: True
```

The raw coastal TIFFs have complete raster-mask coverage, while model support
excludes the official `Fill-NoData` EVT class. The radius-dependent decline is
therefore expected modeled-class truncation where coastal neighborhoods extend
over source NoData ocean, not a seam, reprojection, or AOI-placement error.
Both previews and all automated grid, COG, range, VRT-order, sparse-band, and
class-closure checks passed. The combined planner, raster-engine, Annual NLCD,
and LANDFIRE regression suite passes all 35 tests. The next LANDFIRE gate is
the annual Dist20-Dist23 derivation, followed by bounded LF2016/LF2022 release
checks before a full NC materialization.

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

### 2026-07-20

- Treat official-source header validation as a distinct gate between source
  registration and expensive derivation.
- Prefer requester-pays AWS window reads for scalable state/CONUS processing
  when credentials and the generated object paths validate; retain immutable
  local ZIP/TIFF registration as the supported fallback.
- Define the current land-cover fragmentation band narrowly as one minus the
  dominant class fraction. Do not describe it as patch-edge fragmentation.
- Require every Annual NLCD derivation to match its analytically expected band
  count before writing a successful summary.
- Keep source coverage as an explicit QA band rather than converting
  insufficient support into ecological zeroes.
- Do not claim the Annual NLCD adapter is promoted from synthetic tests alone;
  real NC coverage, seams, ranges, and coastal/island geography remain required.
- Treat the official-source interior pilot as passed: 244/244 bands and COGs,
  151.28 MiB, complete source coverage, valid ranges, and maximum class-fraction
  sum error `7.15e-07`.
- Require a second bounded pilot on `xp0017_yp0014` before the full-state run so
  barrier islands, sounds, ocean-adjacent support, and AOI edge masking are
  tested rather than inferred from an interior tile.
- Keep derived-raster validation reproducible through
  `validate-nlcd-derived`; do not rely on ad hoc array inspection alone.

### 2026-07-21

- Promote the two-tile interior/coastal Annual NLCD gate: 244 bands, 488 COGs,
  196.36 MiB, exact inventory completeness, and no numerical QA issues.
- Treat reduced support at marine-likely checklist locations as declared
  terrestrial-covariate missingness. A plotted traveling-checklist point
  represents the route approximately unless its GPS track is available, so do
  not infer an exact point location for every species detection. Do not impute
  terrestrial land-cover zeroes into open water.
- Accept the 250 m grid for the current NC build because nearly every coastal
  checklist and every inspected Emerald Isle/Fort Macon checklist is supported;
  retain 100 m as a targeted sensitivity, not a new default.
- Proceed to the full 27-tile NC Annual NLCD derivation, then require Ocracoke,
  tidal-wetland, tile-seam, and checklist-location QA before promotion.
- Record elapsed time and COG payload in future derivation summaries so cost
  and runtime comparisons do not depend on console timing or ad hoc inspection.
- Accept the full-state Annual NLCD raster gate: 244 bands, 6,588 COGs,
  2108.33 MiB, 1993.3 seconds, maximum class-fraction sum error `9.54e-07`,
  and no numerical or mapped QA failure across the five inspected tiles.
- Keep Annual NLCD at "provisional pass" until the reusable statewide
  checklist-support diagnostic confirms that unsupported events are expected
  marine/source-edge cases rather than terrestrial extraction gaps.
- Do not accept the first statewide checklist-support percentage at face value:
  164 all-radius failures were valid points inside the AOI but within 98.11 m
  of its boundary, exposing a center-cell raster-mask artifact. The additional
  six 5 km-only failures were marine/pelagic and are expected.
- Use `all_touched` as the regional raster AOI mask contract. Keep exact vector
  AOI membership separate, retain both mask-rule cell counts in plans, record
  the selected rule in derived artifacts, and reject reuse across mask rules.
- Require one full Annual NLCD overwrite after this contract migration, then
  rerun numerical, mapped, and statewide checklist-support QA before promotion.
- Accept the shared AOI mask regression gate: all 14 focused planner, raster,
  catalog, derivation, stale-output, and checklist-support tests passed in
  2.744 seconds.
- Accept the revised all-touched build plan: 2,235,542 selected AOI cells,
  5,295 (`0.2374%`) above the center-rule count, with the same 27 tiles,
  bounding grid, 17 source blocks, and 1,917 estimated logical bands.
- Accept completion of the corrected full overwrite: 244 bands, 27 tiles,
  6,588 COGs, 2,112.22 MiB, and 2,227.7 seconds. The summary records
  `all_touched`; promotion still requires rerunning raster and checklist QA.
- Accept the corrected numerical and mapped raster gate: 244 bands, 6,588
  COGs, maximum class-fraction sum error `9.54e-07`, and all automated checks
  passed. Seven representative previews preserve coherent interior, boundary,
  barrier-island, estuarine, and open-water structure without visible seams.
  Treat lower large-radius support in ocean-heavy eastern tiles as declared
  terrestrial-source truncation, not a derivation failure or a value to impute.
- Promote Annual NLCD after corrected statewide checklist QA. The all-touched
  rebuild reduced all-radius failures from 164 to one and 5 km failures from
  170 to seven. Every remaining event is marine or marine-likely; preserve its
  missing terrestrial support rather than imputing land-cover zeroes.

### 2026-07-22

- Treat LANDFIRE vegetation as periodic rather than annual. Expand the plan by
  explicit source releases and record the release selected for each
  observation year.
- Use LF2016 as the transparent non-future fallback for 2020-2021 while LF2020
  vegetation remains archived and unavailable through the current LFPS
  inventory. Preserve 4- and 5-year source ages as model/provenance fields.
- Use exact annual disturbance content for 2020-2023 even though Dist21 and
  Dist22 are distributed under LF2022.
- Use official public ArcGIS ImageServers for the bounded raster pilot. The
  catalog path has no AWS request charge and does not submit asynchronous LFPS
  jobs.
- Require release-specific class semantics before derivation. Use nearest
  neighbor for categorical source reads, then derive model-grid fractions;
  never bilinearly interpolate EVT/EVC/EVH codes.
- Use each official ImageServer's named `rasterAttributeTable` response for
  release-specific class semantics. Do not depend on byte-range access to
  full-extent ZIPs whose server fails at absolute offsets above 2 GiB.
- Use a nine-class portable EVT hierarchy based on `EVT_PHYS`, preserving all
  official raw fields. Use `EVT_LF` only to disambiguate
  `Exotic Tree-Shrub`; fail on unmapped future categories.
- Treat EVC and EVH values as cover and height conditional on the mapped
  dominant lifeform. Do not present the composite products as simultaneous
  tree, shrub, and herb measurements.
- Accept the first bounded raw-raster gate on `xp0014_yp0015`: LF2023 EVT,
  EVC, and EVH are each one-band 30 m rasters in `EPSG:5070`, have complete
  source coverage, contain only release-registered values, and total
  78.86 MiB.
- Accept the matching coastal raw-raster gate on `xp0017_yp0014`: all three
  products again have complete source coverage and release-registered values.
  Treat the contrast with the seven Annual NLCD marine gaps as a
  source-support difference, not a grid-placement failure.
- Preserve a release's complete logical schema even when a tile-level band is
  all NoData; omit the empty COG but retain an empty inventory and VRT band.
- Accept the bounded LF2023 model-scale vegetation gate. Interior and coastal
  pilots retain 46 logical bands, pass range/grid/VRT and EVT class-closure QA,
  and have maximum fraction-sum error `7.15e-07`. Coastal support declines
  from 99.04% at 250 m to 90.43% at 5 km because modeled neighborhoods reach
  official `Fill-NoData` ocean cells; preserve this as explicit support.
- Proceed to annual disturbance and LF2016/LF2022 bounded gates before any
  full-state LANDFIRE materialization or model refit.

## Open Questions

1. Whether Ocracoke or a later narrow-island transfer test exposes a specific
   need for a targeted 100 m coastal sensitivity; the current coastal pilot
   does not.
2. Whether the conditional dominant-lifeform EVC/EVH summaries add enough
   independent structural information beyond EVT and Annual NLCD to remain in
   the core materialization profile.
3. How to crosswalk 3DHP and legacy NHDPlus HR attributes into one schema.
4. Which NWI Cowardin hierarchy best balances ecological meaning and sparsity.
5. Whether the NC materialized output should include all monthly climate bands
   or whether they should remain in the logical VRT and date-aware extractor.
6. Which second region provides the strongest portability test without
   requiring a full CONUS eBird build.

## Next Ledger Update

The next update should record:

- the exact annual LANDFIRE disturbance derivation and validation result
- bounded LF2016 and LF2022 vegetation release results
- the decision and command for full-NC LANDFIRE materialization
- named materialization profiles and exact inclusion rules
