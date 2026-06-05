# Neural Poisson Point Process

Neural Inhomogeneous Poisson Point Process implementation in PyTorch.

## Project Layout

The repository is organized around reproducible data acquisition, exploratory
analysis, and generated outputs:

- `scripts/data/`: data acquisition scripts for eBird, GBIF, OpenTopography,
  and the R `spatstat.data` export.
- `exp/`: exploratory analysis scripts, including the initial neural
  IPPP analysis.
- `data/`: downloaded or exported tabular/raster data.
- `images/`: generated figures used by the analysis and this README.
- `documents/`: dataset-specific result summaries and interpretation notes.

Commands below assume they are run from the project root.

## Data Preparation Commands

Point Process Datasets:

```
Rscript scripts/data/load-spatstat-data.R
python scripts/data/gbif-anolis-carolinensis-wake-county.py
python scripts/data/ebird-historic-species.py --region US-NC --start 2020-01-01 --end 2020-12-31 --species-code woothr --output data/wood_thrush_nc_2020.csv
python scripts/data/ebird-historic-species.py --region US-NC --start 2021-01-01 --end 2021-12-31 --species-code woothr --output data/wood_thrush_nc_2021.csv
python scripts/data/ebird-historic-species.py --region US-NC --start 2022-01-01 --end 2022-12-31 --species-code woothr --output data/wood_thrush_nc_2022.csv
python scripts/data/ebird-historic-species.py --region US-NC --start 2023-01-01 --end 2023-12-31 --species-code woothr --output data/wood_thrush_nc_2023.csv
python scripts/data/combine-ebird-geojson.py --inputs data/wood_thrush_nc_2020.csv data/wood_thrush_nc_2021.csv data/wood_thrush_nc_2022.csv data/wood_thrush_nc_2023.csv --boundary data/boundaries/nc_state_boundary.gpkg --crs EPSG:3857 --output data/wood_thrush_nc_2020_2023.geojson
python exp/wood_thrush_nippp.py --input data/wood_thrush_nc_2020_2023.geojson --boundary data/boundaries/nc_state_boundary.gpkg --analysis-crs EPSG:5070 --plot-crs EPSG:4326 --epochs-constant 3000 --epochs-linear 10000
```

North Carolina Rasters:

```
python scripts/data/usgs-nc-state-boundary.py
python scripts/data/opentopography-dem-bbox.py --south 33.85116926668266 --north 36.5881334409244 --west -84.32178200052 --east -75.45981513195132 --output data/nc_usgs30m.tif --boundary data/boundaries/nc_state_boundary.gpkg
python scripts/data/usfs-tcc-canopy-bbox.py --south 33.85116926668266 --north 36.5881334409244 --west -84.32178200052 --east -75.45981513195132 --start-year 2020 --end-year 2023 --output data/nc_tcc_2020_2023.tif --boundary data/boundaries/nc_state_boundary.gpkg
python scripts/data/reproject-raster-to-template.py --input data/nc_usgs30m.tif --template data/nc_tcc_2020_2023.tif --output data/nc_usgs30m_match_tcc.tif
python scripts/data/usgs-hydrography.py --south 33.85116926668266 --north 36.5881334409244 --west -84.32178200052 --east -75.45981513195132 --template data/nc_tcc_2020_2023.tif --output data/nc_hydro_distance_match_tcc.tif --boundary data/boundaries/nc_state_boundary.gpkg --search-buffer 10000
python scripts/data/fill-tcc-raster.py --tcc data/nc_tcc_2020_2023.tif --hydro data/nc_hydro_distance_match_tcc.tif --output data/nc_tcc_2020_2023_filled.tif --water-distance-threshold 30 --overwrite
python scripts/data/stack-rasters.py --inputs data/nc_tcc_2020_2023_filled.tif data/nc_usgs30m_match_tcc.tif data/nc_hydro_distance_match_tcc.tif --crs EPSG:3857 --boundary data/boundaries/nc_state_boundary.gpkg --resampling nearest bilinear bilinear --mask-tcc-above 100 --valid-footprint intersection --output data/nc_covariate_stack.tif --overwrite
python scripts/data/join-ebird-raster-covariates.py --points data/wood_thrush_nc_2020_2023.geojson --raster data/nc_covariate_stack.tif --output data/wood_thrush_nc_2020_2023_covariates.geojson --overwrite
python exp/plot_raster_previews.py 
```

eBird Data:

```
python scripts\data\preprocess-ebird-bulk.py --ebd-dir data\ebird\ebd_US-NC_202001_202312_smp_relApr-2026 --output-dir data\ebird\processed_nc_2020_2023 --raster data\nc_covariate_stack.tif --boundary data\boundaries\nc_state_boundary.gpkg --stationary-distance zero --drop-missing-raster-covariates any --overwrite

python scripts/data/summarize-geoparquet.py data/ebird/processed_nc_2020_2023/checklists.geoparquet
```

## Experiments

Long-Leaf Pine Dataset Experiment:

```
python exp/nippp.py
```

Wood Thrush, North Carolina Experiment (static data, no covariates):

```
python exp/wood_thrush_nippp.py --input data/wood_thrush_nc_2020_2023.geojson --boundary data/boundaries/nc_state_boundary.gpkg --analysis-crs EPSG:5070 --plot-crs EPSG:4326 --image-dir images/wood_thrush_nippp_spatial --no-temporal --cv-blocks-per-dim 5 --cv-folds 5 --simulation-count 500 --k-radii 50 --epochs-nonlinear 10000 --hidden-dim 16 --hidden-layers 1 --dropout 0.10 --nonlinear-lr 5e-4 --nonlinear-weight-decay 1e-3
```

Wood Thrush, North Carolina Experiment (temporal data, no covariates):

```
python exp/wood_thrush_nippp.py --input data/wood_thrush_nc_2020_2023.geojson --boundary data/boundaries/nc_state_boundary.gpkg --analysis-crs EPSG:5070 --plot-crs EPSG:4326 --image-dir images/wood_thrush_nippp_temporal --cv-blocks-per-dim 5 --cv-folds 5 --simulation-count 500 --k-radii 50 --epochs-nonlinear 10000 --hidden-dim 16 --hidden-layers 1 --dropout 0.10 --nonlinear-lr 5e-4 --nonlinear-weight-decay 1e-3 --temporal-bins 12 --plot-day-of-year 150

python exp/wood_thrush_temporal_gifs.py --input data/wood_thrush_nc_2020_2023.geojson --boundary data/boundaries/nc_state_boundary.gpkg --analysis-crs EPSG:5070 --plot-crs EPSG:4326 --output-dir images/wood_thrush_nippp_temporal/gifs --models linear nonlinear --grid-size 100 --plot-grid-size 120 --epochs-linear 10000 --epochs-nonlinear 10000 --hidden-dim 16 --hidden-layers 1 --dropout 0.10 --nonlinear-lr 5e-4 --nonlinear-weight-decay 1e-3 --temporal-bins 12
```

Wood Thrush, North Carolina Experiment (static data, with covariates):

```
python exp/wood_thrush_nippp.py --input data/wood_thrush_nc_2020_2023_covariates.geojson --boundary data/boundaries/nc_state_boundary.gpkg --analysis-crs EPSG:5070 --plot-crs EPSG:4326 --image-dir images/wood_thrush_nippp_covariates --no-temporal --covariate-raster data/nc_covariate_stack.tif --covariates canopy_median nc_usgs30m_match_tcc distance_to_waterbody_m distance_to_coastline_m --cv-blocks-per-dim 5 --cv-folds 5 --simulation-count 500 --k-radii 50 --epochs-nonlinear 10000 --hidden-dim 16 --hidden-layers 1 --dropout 0.10 --nonlinear-lr 5e-4 --nonlinear-weight-decay 1e-3
```

Wood Thrush, North Carolina Experiment (temporal data, with covariates):

```
python exp/wood_thrush_nippp.py --input data/wood_thrush_nc_2020_2023_covariates.geojson --boundary data/boundaries/nc_state_boundary.gpkg --analysis-crs EPSG:5070 --plot-crs EPSG:4326 --image-dir images/wood_thrush_nippp_temporal_covariates --covariate-raster data/nc_covariate_stack.tif --covariates canopy_median nc_usgs30m_match_tcc distance_to_waterbody_m distance_to_coastline_m --cv-blocks-per-dim 5 --cv-folds 5 --simulation-count 500 --k-radii 50 --epochs-nonlinear 10000 --hidden-dim 16 --hidden-layers 1 --dropout 0.10 --nonlinear-lr 5e-4 --nonlinear-weight-decay 1e-3 --temporal-bins 12 --plot-day-of-year 150
```

Wood Thrush, North Carolina Experiment (temporal data, with covariates); Penalized run:

```
python exp/wood_thrush_nippp.py --input data/wood_thrush_nc_2020_2023_covariates.geojson --boundary data/boundaries/nc_state_boundary.gpkg --analysis-crs EPSG:5070 --plot-crs EPSG:4326 --image-dir images/wood_thrush_nippp_temporal_covariates_penalized --covariate-raster data/nc_covariate_stack.tif --covariates canopy_median nc_usgs30m_match_tcc distance_to_waterbody_m distance_to_coastline_m --linear-covariate-l2 10 --linear-xy-l2 1 --linear-temporal-l2 0.1 --cv-blocks-per-dim 5 --cv-folds 5 --simulation-count 500 --k-radii 50 --epochs-nonlinear 10000 --hidden-dim 16 --hidden-layers 1 --dropout 0.10 --nonlinear-lr 5e-4 --nonlinear-weight-decay 1e-3 --temporal-bins 12 --plot-day-of-year 150
```


## Background

This repository develops neural and classical inhomogeneous Poisson point
process models for spatial event data. The initial examples include the
Longleaf pine point pattern from `spatstat.data` and Wood Thrush eBird
observations, but the modeling utilities are intended to generalize to other
presence-only spatial point datasets.

The current experiments compare homogeneous spatial intensity against
inhomogeneous models that allow intensity to vary over space. Some datasets also
include marks, such as DBH for Longleaf pine, which can be modeled
conditionally on observed locations.

## Point Process Framework

Let $W$ denote the observation window and let $s_i = (x_i, y_i)$ denote observed tree locations.

An inhomogeneous Poisson point process is defined by an intensity function:

$$
\lambda(s) \geq 0
$$

where $\lambda(s)$ gives the expected density of events near location $s$.

The IPPP log-likelihood is:

$$
\log L(\lambda)=\sum_{i=1}^{n} \log \lambda(s_i)-\int_W \lambda(u)\,du
$$

The first term rewards high intensity at observed tree locations.  
The second term penalizes total expected intensity over the observation window.

## Assumptions

The spatial point process implementation assumes:

- Nonnegative intensity: $\lambda(s) \geq 0$
- Independent increments across disjoint spatial regions
- A finite expected number of points over the observation window
- No temporal component
- No explicit interaction between points beyond first-order intensity variation

The current spatial model is therefore a first-order intensity model, not a clustering, inhibition, or interaction model.

## Berman-Turner Quadrature Approximation

Rather than directly optimizing the continuous IPPP likelihood, the code uses a Berman-Turner quadrature approximation.

The observation window is divided into grid cells. For each cell:

$$
Y_j \sim \text{Poisson}(w_j \lambda(u_j))
$$

where:

- $Y_j$ is the number of observed points in grid cell $j$
- $u_j$ is the center of grid cell $j$
- $w_j$ is the area of grid cell $j$
- $\lambda(u_j)$ is the fitted intensity at the cell center

The approximate log-likelihood used for fitting is:

$$
\sum_j \left[Y_j \log \lambda(u_j) - w_j \lambda(u_j) \right]
$$

Constants independent of model parameters are omitted.

This converts the IPPP estimation problem into a weighted Poisson-regression-like problem.

## Coordinate Preprocessing

The observed coordinates are standardized before being passed into the PyTorch models:

$$
x^* = \frac{x - \bar{x}}{s_x},
\quad
y^* = \frac{y - \bar{y}}{s_y}
$$

The same transformation is applied to the Berman-Turner grid cell centers and to prediction grids used for plotting.

This improves numerical stability and makes the linear coefficients correspond to standardized coordinate effects.

## Models

### 1. Homogeneous Poisson Point Process

The HPPP assumes constant intensity over the whole observation window:

$$
\lambda(s) = \lambda_0
$$

The maximum likelihood estimate is available in closed form:

$$
\hat{\lambda}_0 = \frac{n}{|W|}
$$

This model serves as the primary spatial baseline.

### 2. Constant Neural IPPP

The constant neural model is a PyTorch version of the HPPP:

$$
\lambda(s) = \exp(\theta)
$$

It has one learnable parameter and is initialized at the HPPP estimate. It is mainly used as a sanity check.

If implemented correctly, its fitted log-likelihood should closely match the closed-form HPPP log-likelihood.

### 3. Linear IPPP

The linear IPPP allows first-order spatial variation in intensity:

$$
\lambda(s) = \exp(\beta_0 + \beta_1 x^* + \beta_2 y^*)
$$

This is a log-linear inhomogeneous Poisson point process.

It is initialized at the HPPP solution by setting:

$$
\beta_1 = 0,
\quad
\beta_2 = 0,
\quad
\beta_0 = \log(\hat{\lambda}_0)
$$

Training then estimates whether a linear spatial trend improves fit over the homogeneous baseline.

### 4. Conditional Mark Model

The conditional mark model describes DBH as a function of location, conditional on the observed tree locations.

The response is standardized log-DBH:

$$
z_i = \frac{\log(m_i) - \overline{\log(m)}}{s_{\log(m)}}
$$

where $m_i$ is the DBH mark for tree $i$.

The model assumes:

$$
z_i \mid s_i \sim \text{Normal}(\mu(s_i), \sigma^2)
$$

with linear conditional mean:

$$
\mu(s_i) = \alpha_0 + \alpha_1 x_i^* + \alpha_2 y_i^*
$$

This gives a conditional marked point process decomposition:

$$
\log L_{\text{joint}} = \log L_{\text{spatial}} + \log L_{\text{marks} \mid \text{locations}}
$$

The mark model does not change the fitted spatial intensity. It models DBH variation among trees after conditioning on the observed locations.

## Optimization

The PyTorch models are optimized with Adam using the negative Berman-Turner log-likelihood for the spatial models and Gaussian negative log-likelihood for the conditional mark model.

For numerical stability, the linear predictor in the Linear IPPP is clamped inside the model's forward pass before exponentiation. This clamp is active during both training and prediction.

$$
\eta = \beta_0 + \beta_1 x^* + \beta_2 y^*
$$

$$
\lambda(s) = \exp(\text{clamp}(\eta))
$$

This prevents extreme intensity values during optimization.

## Model Comparison

The fitted spatial models are compared using:

1. Log-likelihood: higher log-likelihood indicates better fit.
2. AIC: $AIC = 2k - 2\log L$, where $k$ is the number of fitted parameters.
3. BIC: $BIC = k \log(n) - 2\log L$, where $n$ is the number of observed points.
4. Likelihood ratio test: the linear IPPP is compared against the HPPP using $LR = 2(\log L_{\text{linear}} - \log L_{\text{HPPP}})$.

Because the linear model adds two parameters relative to the HPPP, the likelihood ratio test uses $df = 2$.

## Diagnostics

The experiment scripts include several diagnostics for checking model behavior:

- HPPP and constant neural IPPP comparison as an optimization sanity check.
- AIC, BIC, and likelihood ratio comparisons between spatial models.
- Pearson residual surfaces on the Berman-Turner quadrature grid.
- Berman-Turner grid sensitivity analysis across multiple grid resolutions.
- Spatial block cross-validation for held-out spatial predictive performance.
- Simulation-based total-count diagnostics from the fitted IPPP.
- Approximate inhomogeneous K-function diagnostics for residual clustering.

## Results Documents

Dataset-specific results are kept outside the README so this file stays focused
on project structure and workflow.

- [Longleaf IPPP results](documents/longleaf-results.md)
- [Wood Thrush North Carolina IPPP results](documents/wood-thrush-nc-results.md)

## Current Limitations

The current implementation does not yet model:

- Nonlinear intensity surfaces
- Spatial interaction or clustering directly
- Environmental covariates in the fitted intensity
- Temporal variation in species occurrence or reporting intensity
- Joint dependence between spatial intensity and marks beyond the conditional decomposition

## Train/Test and Validation

The full-window likelihood, AIC, BIC, and likelihood ratio tests are useful for
likelihood-based inference, while spatial block cross-validation is useful for
checking whether spatial trends generalize to held-out regions.

For spatial point pattern data, ordinary random train/test splits can be
misleading because nearby locations may contain similar spatial information. The
implemented spatial block cross-validation divides the observation window into
spatial folds and evaluates held-out IPPP log-likelihood.

## Planned Extensions

Potential next steps include:

- Adding a nonlinear neural network intensity model
- Comparing linear and nonlinear IPPPs with AIC, BIC, and held-out likelihood
- Extending the mark model to nonlinear conditional means or heteroskedastic variance
- Incorporating spatial covariates
- Adding temporal terms and diagnostics for migratory species
