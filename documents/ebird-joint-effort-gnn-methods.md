# eBird Joint Effort and Multi-Species SDM Plan

## Goal

Build a North Carolina eBird 2020-2023 modeling dataset and baseline workflow for
joint species distribution modeling that explicitly represents observer effort.
The first target is a reusable intermediate dataset, not a final model:

- checklist/location-time nodes with effort, temporal, spatial, and environmental features
- species nodes with taxonomic identifiers and summary frequencies
- detection edges connecting species to checklists or aggregated location-time cells

This representation can support both a joint neural point-process model and a
heterogeneous graph neural network for presence-only SDM.

## Modeling Frame

Observed eBird detections are treated as the result of an ecological process and
an observation process:

\[
\lambda_{\text{obs},j}(s,t,z,e)
=
\lambda_{\text{species},j}(s,t,z)
\cdot
q_{\text{effort}}(s,t,e)
\cdot
p_{\text{detect},j}(s,t,e)
\]

where \(j\) is species, \(s\) is location, \(t\) is time, \(z\) are
environmental covariates, and \(e\) are checklist or observer effort variables.

For neural implementation, a practical log-intensity decomposition is:

\[
\log \lambda_{\text{obs},j}
=
f_\theta(j, s, t, z)
+
g_\phi(s, t, e)
+
h_\psi(j, e, t)
\]

- \(f_\theta\): ecological species-environment response
- \(g_\phi\): shared observer effort/reporting surface
- \(h_\psi\): species-specific detectability or reporting bias

This is not identifiable from eBird alone without assumptions. The workflow
should therefore emphasize baselines, ablations, spatial validation, and
sensitivity checks.

## Initial Data Scope

Use the all-species North Carolina eBird Basic Dataset extract:

- region: `US-NC`
- years: 2020-2023
- complete checklists only: `ALL SPECIES REPORTED = 1`
- primary protocols: stationary (`P21`) and traveling (`P22`)
- deduplicate shared checklists by `GROUP IDENTIFIER`
- drop records with invalid coordinates
- restrict to plausible effort ranges

Suggested first effort filters:

- duration minutes: greater than 0 and no more than 300
- traveling distance: no more than 10 km
- number of observers: no more than 20

These filters should be revisited after inspecting retained checklist counts by
protocol, county, year, month, and observer/checklist density.

## Preprocessing Outputs

Create three core tables:

1. `checklists.geoparquet`

   One row per retained checklist or deduplicated checklist group. Geometry is
   the checklist point. Features include:

   - `sampling_event_identifier`
   - `group_identifier`
   - latitude, longitude
   - date, year, month, day of year, day of week
   - time observations started where available
   - observer id where available
   - protocol code and protocol name
   - duration minutes
   - effort distance km
   - effort area ha
   - number observers
   - locality, county, BCR/IBA/USFWS codes
   - optional sampled raster covariates

2. `detections.parquet`

   One row per species-checklist detection edge:

   - `sampling_event_identifier`
   - `taxon_concept_id`
   - `common_name`
   - `scientific_name`
   - `category`
   - parsed `observation_count` where numeric

3. `species.csv`

   One row per species with retained checklist frequency and detection count.
   This table is the initial species-node lookup.

Optional later outputs:

- aggregated location-time cells with counts and checklist effort
- pseudo-negative/background samples for link prediction
- train/validation/test spatial blocks
- observer or checklist-group node tables

## Baselines

Run baselines before the graph/neural model:

1. Single-species Wood Thrush IPPP/NIPPP using existing repository workflow.
2. Multi-species tabular model without graph message passing:

   \[
   \Pr(y_{j,c}=1)=\sigma(f_\theta(x_c, e_j))
   \]

   where \(c\) is checklist and \(e_j\) is a species embedding.

3. Effort-only/shared-bias model using checklist density, protocol, duration,
   distance, observer count, day of week, month, and spatial coordinates.
4. Ecological covariates only.
5. Ecological plus effort covariates.

The graph model should be compared against these to separate gains from
multi-species learning, effort covariates, and graph structure.

## Heterogeneous Graph Design

Start with a bipartite graph:

- node type `species`
- node type `checklist` or `location_time`
- edge type `detected_on`

Candidate extensions:

- `observer` nodes connected to checklists
- `locality` or hotspot nodes connected to checklists
- `county` or spatial-block nodes
- `species_taxonomy` edges if useful trait or taxonomic data are added
- `location_neighbor` edges based on spatial adjacency or distance
- temporal edges between repeated visits to the same locality or grid cell

For first implementation, checklist nodes are easier because eBird effort
metadata are attached directly to checklists. Location-time cells may become
useful once the raw checklist graph is too large or too sparse.

## Training Objective Options

For graph link prediction:

- positive edges are observed species-checklist detections
- negative edges are sampled unobserved species on complete checklists
- negatives should be sampled within plausible candidate species pools
- loss should be weighted to control dominance by common species and high-effort
  checklists

For point-process modeling:

- construct quadrature/background samples over space-time
- model checklist effort as exposure or as a separate intensity term
- compare observed species detections against checklist availability

These objectives answer related but different questions. Link prediction on
complete checklists is usually the faster first step.

## Validation

Use blocked validation that tests transfer across observer geography:

- spatial blocks across North Carolina
- strata based on checklist density or observer effort
- temporal holdouts by year or season
- species-stratified metrics so common species do not dominate
- rare-species metrics for species with enough detections

Key metrics:

- held-out checklist species prediction AUROC/AUPRC
- calibration by effort stratum
- performance by county or spatial block
- Wood Thrush-specific comparison against existing IPPP/NIPPP outputs
- sensitivity to protocol and effort filters

## Immediate Implementation Steps

1. Preprocess the bulk EBD and sampling files into checklist, detection, and
   species tables.
2. Sample existing NC raster covariates onto checklist points.
3. Inspect retained counts and missingness.
4. Build a small pilot subset:

   - top 20-100 species by checklist frequency
   - complete stationary/traveling checklists only
   - 2020-2023 all NC

5. Train tabular multi-species baseline.
6. Add graph construction and a simple heterogeneous GNN.
7. Add effort-only and ecology-only ablations.
8. Compare Wood Thrush predictions to existing single-species IPPP/NIPPP runs.
