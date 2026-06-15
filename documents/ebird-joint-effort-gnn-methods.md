# eBird Joint Effort and Multi-Species SDM Plan

## Goal

Build a North Carolina eBird 2020-2023 modeling dataset and baseline workflow for
joint species distribution modeling that explicitly represents observer effort.
The first target is a reusable intermediate dataset, not a final model:

- checklist/location-time nodes with effort, temporal, spatial, and environmental features
- species nodes with taxonomic identifiers and summary frequencies
- detection edges connecting species to checklists or aggregated location-time cells

This representation can support both a joint neural point-process model and a
heterogeneous graph neural network for complete-checklist detection modeling.
Although the broader SDM motivation starts from presence-only citizen-science
data, the current eBird workflow uses complete checklists, so unreported modeled
species on retained checklists are treated as observed non-detections under an
explicit observation/effort process rather than as arbitrary pseudo-absences.

## Glossary

### Core modeling terms

- **Species distribution model (SDM)**: A model relating species observations to
  spatial, temporal, environmental, and sampling variables. In this project the
  immediate target is checklist-level detection; biological occurrence is only
  partially observed through the eBird reporting process.
- **Occupancy / biological presence**: Whether a species truly occurs at a
  location and time. Occupancy is not directly observed in this workflow because
  a complete checklist can still miss present species.
- **Detection probability**: The probability that a species is reported on a
  checklist, conditional on location, time, habitat, observer effort, and the
  reporting process. This is the modeled outcome here, not confirmed abundance.
- **Complete checklist**: An eBird checklist where the observer indicated that
  all species they could identify were reported. For retained complete
  checklists, an unreported modeled species is treated as an observed
  non-detection under the eBird observation process.
- **Detection**: A species-checklist pair where the species was reported.
- **Non-detection**: A species-checklist pair where the checklist is complete
  and the species was not reported. This is not proof that the species was
  truly absent.
- **True absence**: A location/time where a species was actually absent. True
  absences are rarely known in citizen-science data and should not be confused
  with complete-checklist non-detections.
- **Presence-only data**: Data containing reported presences without reliable
  absences or non-detections. The broader SDM motivation comes from
  presence-only citizen-science data, but this workflow uses complete eBird
  checklists to construct detections and non-detections.
- **Pseudo-absence / background sample**: A sampled location or
  species-checklist pair used as a comparison point when true absences are
  unavailable. In this workflow, arbitrary pseudo-absences are mostly avoided,
  but sampled unobserved species-checklist pairs are used in some graph link
  baselines.
- **Ecological process**: The latent biological process governing where and when
  species occur, based on habitat, climate, seasonality, geography, and other
  environmental factors.
- **Observation process**: The process determining whether a species is detected
  and reported, given that it may be present. This includes effort, protocol,
  duration, distance traveled, observer count, time of day, and reporting
  behavior.
- **Observer effort**: Checklist-level variables that affect the chance of
  detecting and reporting species, such as duration, distance traveled,
  protocol type, number of observers, and start time.
- **Protocol**: The eBird sampling method for a checklist. Protocol matters
  because stationary, traveling, area, incidental, and other protocols imply
  different observation processes and different meanings for effort variables.
- **Stationary checklist**: A checklist where the observer sampled from one
  location rather than along a route. In this workflow, stationary effort
  distance should be zero or missing by construction, not inferred as movement.
- **Traveling checklist**: A checklist where the observer sampled while moving
  along a route. Distance traveled is meaningful for these records and can
  affect detection probability.
- **Reporting bias / sampling bias**: Non-ecological structure in where, when,
  and how observers submit checklists. Examples include hotspots, roads, parks,
  weekends, high-effort birding routes, and observer preferences.
- **Observer geography**: Spatial structure caused by where people go birding
  rather than where species occur. It can improve prediction while hurting
  ecological interpretation or transfer.
- **Effort/reporting surface**: A modeled spatial-temporal pattern in checklist
  intensity, observer behavior, or reporting probability that is shared across
  species.
- **Checklist intensity**: The rate or density of checklist submissions over
  space and time. It reflects observation effort, not species occurrence.
- **Species-specific detectability**: Variation in how easily different species
  are detected or reported under the same effort conditions.
- **Identifiability**: Whether model components can be uniquely separated from
  the data. In eBird data, ecological occurrence, effort, and detectability are
  not fully identifiable without assumptions, so ablations and validation are
  essential.
- **Checklist node**: A graph node representing one retained eBird checklist or
  deduplicated checklist group.
- **Species node**: A graph node representing one modeled species.
- **Observer node**: A possible future graph node representing an eBird
  observer. Observer nodes could capture repeated observer behavior or skill,
  but they need careful privacy, leakage, and transfer handling.
- **Locality node / hotspot node**: A possible future graph node representing a
  repeated eBird location or hotspot. These nodes can capture locality history
  but may also encode observer geography too strongly.
- **Detection edge / positive edge**: A graph edge connecting a species node to
  a checklist node when that species was reported on that checklist.
- **Negative edge**: A species-checklist pair where the checklist was complete
  and the species was not reported, or a sampled unobserved pair used for link
  prediction.
- **All-pairs target**: The full target formed by crossing every held-out
  checklist with every modeled species. This is the fair evaluation target for
  checklist-by-species detection prediction.
- **Sampled-edge target**: A training or evaluation target formed from sampled
  positive and negative species-checklist edges, often with artificial
  positive/negative balance. It is useful for graph link diagnostics but is not
  directly comparable to all-pairs evaluation unless prevalence is accounted
  for.
- **Target distribution**: The distribution of examples the model is intended
  to predict on. Here, the main target distribution is all held-out
  checklist-by-species pairs under the spatial-stratified split.
- **Data leakage**: Using information from held-out data during feature
  construction, splitting, calibration, or training. Train-only aggregates and
  leave-one-out adjustments are used to avoid leakage.
- **Train-only aggregate**: A locality, spatial-cell, or neighbor summary
  computed from training checklists only before being applied to held-out
  checklists.
- **Leave-one-out aggregate**: A training-row aggregate that excludes the
  checklist's own labels so a checklist does not copy its target into its
  features.

### Metrics

- **AUROC / ROC AUC**: Area under the receiver operating characteristic curve. It measures how well the model ranks positive examples above negative examples across thresholds. A value of 1.0 is perfect ranking, while 0.5 is random ranking.
- **AUPRC / PR AUC**: Area under the precision-recall curve. It measures the
  tradeoff between precision and recall across thresholds. AUPRC is especially
  useful when detections are rare because its baseline depends on prevalence.
  AUPRC values are only directly comparable when evaluated on the same target
  distribution.
- **Precision**: Among the species-checklist pairs predicted as positive, the proportion that were actually positive.
- **Recall**: Among the actually positive species-checklist pairs, the proportion the model identified as positive.
- **Macro AUROC / Macro AUPRC**: The metric is computed separately for each species and then averaged across species. Macro metrics treat each species equally, so they are useful for checking whether performance gains help uncommon species rather than only common species.
- **Micro AUROC / Micro AUPRC**: The metric is computed after pooling all species-checklist pairs together. Micro metrics weight common species and common detection patterns more heavily because they contribute more observations.
- **Train AUROC / Train AUPRC**: AUROC or AUPRC computed on the training data. These metrics help diagnose whether the model is fitting the training target.
- **Test AUROC / Test AUPRC**: AUROC or AUPRC computed on held-out data. These metrics are the primary indicators of generalization.
- **Species macro AUROC / Species macro AUPRC**: Species-level AUROC or AUPRC averaged across modeled species. This is used in graph-link outputs to distinguish species-level performance from pooled sampled-edge performance.
- **Prevalence**: The observed proportion of positive species-checklist pairs. Prevalence can be computed per species, across all pairs, or within a sampled training or evaluation set.
- **All-pairs prevalence**: The detection prevalence across the full
  checklist-by-species target. This is much lower than a balanced sampled-edge
  prevalence and is the relevant prevalence for all-pairs calibration.
- **Sampled-edge prevalence**: The positive fraction in a sampled edge training
  or evaluation set. It may be artificial, such as 50/50 positive/negative, and
  can make sampled-edge AUPRC and raw probabilities misleading for all-pairs
  evaluation.
- **Train prevalence baseline**: A baseline that predicts each species' training prevalence for every checklist. It contains no checklist-level ecological or effort information. Its macro AUROC is 0.5 because it gives the same score to every checklist within a species. Its micro AUROC can exceed 0.5 because different species receive different constant scores.
- **Probability-bin calibration**: A calibration check where predictions are grouped into probability bins and compared with observed frequencies within each bin.
- **ECE / Expected Calibration Error**: The weighted average absolute difference between predicted probability and observed frequency across probability bins. Lower values indicate better aggregate probability calibration.
- **Max bin error**: The largest absolute calibration error among probability bins. It identifies the worst-calibrated probability range, even when average ECE is low.
- **Species calibration error**: For one species, the difference between that species' mean predicted detection probability and its observed detection rate.
- **Species calibration MAE**: The mean absolute species calibration error across species. Lower values indicate that predicted probabilities are better aligned with species-level observed detection rates.
- **Effort-stratum calibration**: Calibration measured within effort-defined groups, such as duration bins, traveling-distance bins, protocol types, or observer-count bins.
- **Ranking metric**: A metric such as AUROC or AUPRC that evaluates ordering of predictions, not whether predicted probabilities are calibrated.
- **Calibration metric**: A metric such as ECE or species calibration MAE that evaluates whether predicted probabilities match observed frequencies.

### Baseline and tabular models

- **Prevalence baseline**: A constant-probability baseline where each species is assigned its training detection prevalence. It is a sanity-check reference for more complex models.
- **Linear ecology model**: A logistic regression model using environmental, spatial, and temporal covariates, but not effort variables. It tests how much predictive signal comes from ecological covariates alone.
- **Linear effort model**: A logistic regression model using checklist and observer-effort variables, but not environmental covariates. It tests how much predictive signal comes from the observation/reporting process alone.
- **Linear ecology + effort model**: A logistic regression model combining ecological and effort covariates. It tests whether ecological and observation-process variables provide complementary predictive signal.
- **MLP**: Multilayer perceptron. A feed-forward neural network that can learn nonlinear relationships among covariates.
- **MLP ecology model**: An MLP using ecological covariates only.
- **MLP effort model**: An MLP using effort covariates only.
- **MLP ecology + effort model**: An MLP using both ecological and effort covariates. In the current workflow, this is the strongest non-graph tabular baseline.
- **Tabular joint baseline**: A multi-species model that predicts checklist-level detections from tabular checklist features and species information, without graph message passing.
- **Species embedding**: A learned vector representation of a species. Species embeddings let a joint model share information across species while still learning species-specific detection patterns.
- **Joint multi-species model**: A model that predicts detections for multiple species in one shared framework, rather than fitting a separate model for each species.

### Point-process models

- **Point process**: A statistical model for events observed at locations in space or space-time. In species modeling, the events are usually species occurrence or detection locations.
- **Intensity**: The expected rate of points per unit area or space-time. Higher intensity means the model expects more observations in that region.
- **Exposure / offset**: A known or separately estimated term that scales an
  expected count or intensity. In this project, checklist effort could sometimes
  be treated as an exposure-like quantity, but only with care because effort
  also changes detectability and reporting behavior.
- **IPP / Inhomogeneous Poisson process**: A point-process model where the intensity varies across space, time, or covariates.
- **IPPP**: Inhomogeneous Poisson point process. This term is often used interchangeably with IPP in spatial point-process modeling.
- **NIPPP / Neural IPPP**: A neural inhomogeneous Poisson point process where a neural network represents the log-intensity function. It can model nonlinear species-environment relationships.
- **Single-species IPPP / NIPPP**: A point-process model fit for one species at a time. In this workflow, this serves as a single-species spatial modeling reference before moving to joint multi-species models.
- **Log-intensity decomposition**: A modeling structure where observed intensity is decomposed into ecological, effort/reporting, and species-specific detectability components on the log scale.
- **Quadrature / integration points**: Background locations used to approximate
  the spatial integral in a point-process likelihood. They are not observed
  absences; they are numerical integration support for the likelihood.

### Graph and bridge models

- **Heterogeneous graph**: A graph with multiple node or edge types. In this workflow, the main node types are species and checklists, and the main edge type is detected-on.
- **Bipartite graph**: A graph with two node sets where edges connect nodes across sets, not within the same set. The species-checklist detection graph is bipartite because edges connect species to checklists.
- **Graph neural network / GNN**: A neural network that learns from graph-structured data by passing information between connected nodes.
- **Message passing**: The process by which a GNN updates node representations using information from neighboring nodes.
- **Species co-detection / co-occurrence signal**: Information from species that
  are reported together on checklists. It can help joint prediction, but it is
  still an observation-level signal and should not be interpreted automatically
  as biological interaction.
- **Non-message-passing link baseline**: A link-prediction model that uses species embeddings and checklist features to predict species-checklist detections, but does not pass messages across graph edges.
- **Sampled-edge link baseline**: A graph link model trained on sampled positive and negative species-checklist edges. It is useful as a bridge between tabular models and GNNs, but its sampled prevalence may differ from the real all-pairs target prevalence.
- **All-species checklist-batch bridge**: A non-message-passing bridge model trained by scoring all modeled species for each checklist in a batch. This aligns training with the all-pairs checklist-by-species evaluation target.
- **Pair MLP bridge**: A bridge model that concatenates checklist features with a species embedding and passes the pair through an MLP to score each species-checklist pair.
- **Factorized bridge**: A bridge model that encodes each checklist into a latent vector, then scores species using a low-rank interaction such as a dot product with species embeddings plus checklist and species biases.
- **Hybrid bridge**: A bridge model that combines a direct multi-species checklist prediction head with a species-embedding interaction term. This retains the strength of the tabular MLP while adding explicit species-embedding structure.
- **Stronger hybrid bridge**: A higher-capacity hybrid bridge with larger hidden dimensions, more hidden layers, and larger latent species/checklist interaction space. In the current workflow, this becomes a strong non-GNN bridge baseline.
- **Link prediction**: The task of predicting whether an edge exists between two nodes. Here, it means predicting whether a species was detected on a checklist.

### Spatial and relational model components

- **Spatial block holdout**: A validation strategy that holds out spatial blocks rather than randomly holding out checklists. This tests whether the model generalizes to new areas rather than memorizing repeated locations.
- **Spatial-stratified split**: A spatial validation split designed to hold out blocks while approximately preserving checklist effort, environmental covariates, and common-species prevalence.
- **Spatial transfer**: Generalization from training areas to held-out areas.
  This is central to the bias/effort goal because a model that only memorizes
  observer geography may score well on random splits but fail under spatial
  transfer.
- **Locality / hotspot**: A recurring eBird location where multiple checklists may be submitted over time.
- **Locality/spatial prior**: A train-only prior based on species detection rates at localities or spatial cells. It can encode useful local information but can also overfit sampling geography.
- **Prior logit**: A prior probability transformed to the logit scale and added to model logits. In this workflow, locality or spatial species-detection rates may be converted into prior logits.
- **Prior-logit weight**: A learned scalar weight controlling how strongly the model uses the prior logit. Initializing this weight at zero forces the model to learn whether the prior is useful instead of relying on it immediately.
- **Spatial-neighbor scalar features**: Checklist-level features summarizing nearby training checklist density or detection rates using distance-weighted neighboring spatial cells.
- **Spatial-neighbor prior logits**: Species-specific prior logits computed from nearby training cells rather than the same locality or same grid cell. These are smoother than direct locality/spatial-cell priors.
- **RBF spatial residual**: A smooth spatial correction term based on radial basis functions. It models residual spatial structure left over after ecology, effort, temporal variables, and species interactions are accounted for.
- **Spatial residual**: A model component added to capture leftover spatial pattern in predictions. It can improve ranking but may also encode observer geography or sampling bias if not carefully constrained.
- **Spatial-cell graph**: A graph where nodes represent spatial grid cells and edges connect neighboring cells.
- **Queen-neighbor edges**: Spatial adjacency edges connecting grid cells that touch by either an edge or a corner, similar to queen contiguity in spatial analysis.
- **GCN / Graph convolutional network**: A GNN architecture that updates node embeddings by aggregating information from neighboring nodes.
- **Spatial-cell GCN**: A GCN applied to spatial-cell nodes. It learns spatial context from neighboring cells and passes that context into the species detection model.
- **Concat spatial-cell GCN**: A GNN design that concatenates each checklist's spatial-cell embedding with checklist features before prediction. This directly injects message-passing context into the detector.
- **Residual spatial-cell GCN**: A GNN design that keeps the stronger checklist/species model as the base prediction and adds a small species-specific spatial GCN correction to the logits.
- **Gated residual spatial-cell GCN**: A residual GNN design where the spatial correction is multiplied by a learned gate. The gate allows the model to reduce or ignore spatial messages when they hurt transfer or calibration.
- **Over-smoothing**: A GNN failure mode where repeated message passing makes neighboring node representations too similar, reducing useful local or species-specific signal.

### Calibration and prior correction

- **Logit**: The log-odds transformation of a probability: `log(p / (1 - p))`.
- **Sigmoid**: The inverse-logit function that maps logits back to probabilities between 0 and 1.
- **Case-control prior correction**: A post-hoc correction for models trained on artificially balanced positive/negative samples. It shifts logits from the sampled training prevalence to the target all-pairs prevalence.
- **Global prior correction**: A single logit shift applied to all predictions to adjust for the difference between sampled-edge prevalence and all-pairs prevalence.
- **Species-specific intercept calibration**: A calibration method that adjusts each species' predictions with its own intercept shift.
- **Platt scaling**: A calibration method that fits a logistic transformation of model scores on a calibration set.
- **Calibration split**: A held-out subset used to fit calibration adjustments without using the test set.

### Model comparison terms

- **Ablation**: A controlled model comparison where one component or feature group is removed or changed. For example, comparing ecology-only, effort-only, and ecology + effort models isolates the contribution of each feature group.
- **Feature set**: The group of input variables used by a model, such as ecology, effort, or both.
- **Ecology covariates**: Environmental, spatial, or temporal variables intended to represent species habitat or ecological conditions.
- **Effort covariates**: Checklist and observer variables intended to represent the observation/reporting process.
- **Temporal covariates**: Time-derived variables such as day of year, day of
  week, month, year, or start time. They can encode seasonality and observation
  behavior.
- **Environmental covariates**: Raster or tabular variables describing
  environmental conditions, such as canopy cover, elevation, land cover, or
  distance to water.
- **Bridge model**: A model between tabular baselines and full GNNs. It uses graph-ready species/checklist data structures and species embeddings, but may not use graph message passing.
- **Relational-feature baseline**: A non-GNN baseline that adds graph-like information, such as locality history or spatial-neighbor summaries, as explicit features.
- **Fair comparison target**: An evaluation target that matches the intended prediction task. In this workflow, all held-out checklist-by-species pairs are the fair comparison target for tabular and graph models.
- **Target-aware objective**: A training objective that matches the intended evaluation distribution. The all-species checklist-batch bridge is target-aware because it trains on the full checklist-by-species label matrix rather than a balanced sampled-edge set.
- **Model capacity**: The flexibility of a model, controlled by factors such as hidden dimension, number of layers, embedding dimension, and architecture.
- **Underfitting**: A model failure mode where the model is too simple to capture important structure in the data.
- **Overfitting**: A model failure mode where the model learns training-specific patterns that do not generalize to held-out data.
- **Regularization**: Techniques such as dropout or weight decay that limit overfitting.
- **Weight decay**: A regularization method that penalizes large model weights during training.
- **Dropout**: A regularization method that randomly disables hidden units during training to reduce dependence on any single pathway.
- **Residual correction**: An additive model component that starts from a base
  prediction and learns only the leftover structure. The spatial residual and
  residual spatial-cell GCN are both residual corrections.
- **Gate**: A learned multiplier that controls how much a residual or message
  passing component can affect a prediction.
- **Species-level diagnostics**: Per-species metrics and calibration checks used to identify which species improve or degrade under a model.
- **Aggregate metrics**: Metrics computed across many species-checklist pairs or averaged across species. Aggregate improvements can hide species-specific failures, so they should be interpreted alongside species-level diagnostics.

## Data Preparation

```
python scripts\data\preprocess-ebird-bulk.py --ebd-dir data\ebird\ebd_US-NC_202001_202312_smp_relApr-2026 --output-dir data\ebird\processed_nc_2020_2023 --raster data\nc_covariate_stack.tif --boundary data\boundaries\nc_state_boundary.gpkg --stationary-distance zero --drop-missing-raster-covariates any --overwrite

python scripts/data/summarize-geoparquet.py data/ebird/processed_nc_2020_2023/checklists.geoparquet
```

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

First multi-species tabular baseline:

```
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 20 --feature-set both --epochs 50
```

Then the two ablations with the same settings:

```
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 20 --feature-set effort --epochs 50
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 20 --feature-set ecology --epochs 50
```

Top-20 held-out 2023 results so far:

| Model | Macro AUROC | Macro AUPRC | Micro AUROC | Micro AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Train prevalence | 0.5000 | 0.3644 | 0.6442 | 0.5064 |
| Linear ecology | 0.6739 | 0.5024 | 0.7240 | 0.5863 |
| Linear effort | 0.7101 | 0.5531 | 0.7468 | 0.6214 |
| Linear ecology + effort | 0.7461 | 0.5954 | 0.7755 | 0.6610 |

Notes from this first baseline:

- Effort features alone outperform ecology-only features, which is consistent
  with strong checklist/reporting effects in complete-list eBird data.
- Ecology-only features still add substantial predictive signal over prevalence.
- Combining ecology and effort improves over both ablations, so the two feature
  groups are complementary rather than redundant.
- The pooled prevalence baseline has micro AUROC above 0.5 because species have
  different constant prevalence scores. Macro AUROC is the within-species sanity
  check and remains 0.5.
- Per-species comparison shows the combined model improves AUPRC over the
  effort-only model for all top-20 species. The largest gains are for
  White-throated Sparrow, Chipping Sparrow, Eastern Bluebird, and Downy
  Woodpecker.

Per-species comparison command:

```
python exp/compare_ebird_tabular_baselines.py --baseline-dir data/ebird/baselines --top-species 20
```

Top-100 held-out 2023 results:

| Model | Macro AUROC | Macro AUPRC | Micro AUROC | Micro AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Train prevalence | 0.5000 | 0.1400 | 0.7725 | 0.3874 |
| Linear ecology | 0.7610 | 0.2880 | 0.8486 | 0.4740 |
| Linear effort | 0.7861 | 0.3225 | 0.8578 | 0.5044 |
| Linear ecology + effort | 0.8148 | 0.3673 | 0.8738 | 0.5432 |

Top-100 notes:

- The same pattern holds after scaling from 20 to 100 species: effort-only beats
  ecology-only, ecology adds signal beyond effort, and the combined model is
  strongest overall.
- Macro AUPRC drops from the top-20 run because the additional species are less
  prevalent. This is expected and makes AUPRC the more useful metric for the
  scale-up check.
- The largest gains of combined over effort-only AUPRC include Red-breasted
  Nuthatch, Brown Pelican, Summer Tanager, Ring-billed Gull, Dark-eyed Junco,
  and Blue-headed Vireo.
- The combined model is slightly worse than effort-only for Pine Siskin and
  Swamp Sparrow on AUPRC. These are useful diagnostics for later model checks,
  especially for irregular irruptive/wetland-associated species where simple
  linear covariate effects may be too limited.

Top-100 commands used:

```
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --feature-set effort --epochs 50
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --feature-set ecology --epochs 50
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --feature-set both --epochs 50
python exp/compare_ebird_tabular_baselines.py --baseline-dir data/ebird/baselines --top-species 100
```

Top-100 spatial-stratified held-out block results:

| Model | Macro AUROC | Macro AUPRC | Micro AUROC | Micro AUPRC |
| --- | ---: | ---: | ---: | ---: |
| Train prevalence | 0.5000 | 0.1370 | 0.7638 | 0.3848 |
| Linear ecology | 0.7590 | 0.2840 | 0.8467 | 0.4876 |
| Linear effort | 0.7887 | 0.3275 | 0.8563 | 0.5040 |
| Linear ecology + effort | 0.8216 | 0.3737 | 0.8762 | 0.5568 |

Spatial validation notes:

- The spatial-stratified holdout selected 3 of 48 populated blocks and held out
  20.0% of checklists. The held-out blocks were selected to preserve overall
  checklist effort, environmental covariates, and common-species prevalence as
  much as possible.
- Results are very close to, and slightly stronger than, the temporal top-100
  split. This suggests the current linear features are not only memorizing the
  repeated 2020-2022 geography when evaluated on 2023.
- Combined ecology + effort remains the strongest model under spatial
  validation. Effort-only still outperforms ecology-only overall, but ecology
  adds meaningful complementary signal.
- The combined model improves over effort-only AUPRC for nearly all top-100
  species. Exceptions are Yellow-rumped Warbler, Swamp Sparrow, Eastern
  Meadowlark, Royal Tern, and Ovenbird, where effort-only is slightly better.

Top-100 spatial-stratified calibration:

| Model | Probability-bin ECE | Max bin error |
| --- | ---: | ---: |
| Train prevalence | 0.0082 | 0.0336 |
| Linear ecology | 0.0063 | 0.2404 |
| Linear effort | 0.0096 | 0.0333 |
| Linear ecology + effort | 0.0126 | 0.0362 |

Calibration notes:

- Probability-bin expected calibration error is low for all models, so aggregate
  predicted probabilities are broadly sane at the species-checklist-pair level.
- The combined model remains the best ranking model by AUROC/AUPRC, but it is
  not the best calibrated by aggregate ECE. This is a useful distinction for
  the effort-modeling goal.
- Ecology-only has the lowest ECE but a high max bin error, suggesting most
  predictions are calibrated on average while one predicted-probability bin is
  substantially off.
- Effort-only and combined have similar max bin errors to the prevalence
  baseline, but combined has the largest ECE. A later calibration step should
  inspect the `*_calibration.csv` rows by protocol, duration, effort distance,
  and observer count rather than relying only on aggregate probability bins.

Calibration comparison command:

```
python exp/compare_ebird_calibration.py --top-species 100 --split spatial-stratified
```

Effort-stratum calibration findings:

- Ecology-only has the largest effort-stratum calibration errors, as expected
  because it has no direct protocol, duration, distance, or observer-count
  inputs. It underpredicts high-effort checklists such as `121+` minute
  checklists and overpredicts short `1-10` minute checklists.
- Effort-only and combined largely correct the duration/protocol calibration
  pattern. Their largest remaining effort-stratum issue is overprediction for
  `5+ km` traveling-distance checklists.
- For effort-only, the `5+ km` stratum has mean predicted detection probability
  0.2333 vs observed 0.1946. For combined, the same stratum is 0.2280 vs 0.1946.
- Combined remains the best ranking model, but effort-only is slightly better
  calibrated across several effort strata. This supports keeping calibration
  diagnostics separate from AUROC/AUPRC when evaluating bias/effort modeling.
- Ecology-only has one severe high-probability bin issue: the `(0.9, 1.0]`
  predicted-probability bin has mean predicted probability 0.9204 vs observed
  0.6800, though it contains only 100 species-checklist pairs.

Top-100 spatial-stratified MLP results:

| Model | Macro AUROC | Macro AUPRC | Micro AUROC | Micro AUPRC | ECE | Max bin error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP ecology | 0.7925 | 0.3257 | 0.8654 | 0.5115 | 0.0055 | 0.0573 |
| MLP effort | 0.8283 | 0.3913 | 0.8825 | 0.5681 | 0.0087 | 0.0373 |
| MLP ecology + effort | 0.8444 | 0.4194 | 0.8921 | 0.5864 | 0.0050 | 0.0219 |

MLP notes:

- The small MLP improves over the linear baseline for all feature sets, so some
  of the remaining baseline gap was linear underfitting rather than graph
  structure.
- The MLP ecology + effort model is now the best model by both ranking metrics
  and aggregate probability-bin calibration.
- Relative to the linear ecology + effort model, MLP ecology + effort improves
  macro AUPRC from 0.3737 to 0.4194 and micro AUPRC from 0.5568 to 0.5864.
- MLP ecology-only still shows the largest effort-stratum calibration errors,
  especially `121+` minute and `1-10` minute checklists. Adding effort features
  remains necessary even with nonlinear covariate effects.
- MLP ecology + effort reduces the remaining `5+ km` overprediction problem:
  mean predicted probability is 0.2120 vs observed 0.1946, compared with 0.2280
  for the linear ecology + effort model.
- Several species benefit more from MLP ecology + effort than MLP effort,
  especially coastal/water-associated species such as Great Black-backed Gull,
  Brown Pelican, Ring-billed Gull, American Herring Gull, Killdeer, Great
  Egret, and swallows. Some upland/generalist species show little or negative
  gain over MLP effort, so species-level diagnostics remain important.

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

Before training a graph model, freeze a graph-ready dataset format using the
same top-100 species set and the same spatial-stratified split as the tabular
baselines. The first graph dataset should contain:

- `checklist_nodes.parquet`: one row per checklist with checklist id, raw effort
  fields, temporal fields, spatial coordinates, split masks, and feature row
  index.
- `checklist_features.npy`: standardized checklist feature matrix, fit using
  training checklists only.
- `species_nodes.csv`: one row per modeled species with species index,
  taxonomic metadata, and detection frequency.
- `positive_edges.parquet`: observed species-checklist detections for the
  modeled top species.
- `negative_edges.parquet`: sampled unobserved species-checklist pairs from
  complete checklists.
- `metadata.json`: feature names, split settings, held-out block ids, negative
  sampling settings, and output row counts.

Graph dataset builder command:

```
python exp/build_ebird_graph_dataset.py --processed-dir data/ebird/processed_nc_2020_2023 --output-dir data/ebird/graph_top100_spatial --top-species 100 --split spatial-stratified --spatial-blocks-per-dim 8 --test-fraction 0.2 --negative-ratio 5
```

Graph dataset validation command:

```
python exp/validate_ebird_graph_dataset.py --graph-dir data/ebird/graph_top100_spatial
```

This step deliberately does not train a GNN. After verifying row counts, split
masks, positive edge counts, and negative sampling balance, the next modeling
step should be a simple non-message-passing species/checklist embedding link
model using this exact dataset.

Non-message-passing graph link baseline command:

```
python exp/ebird_graph_link_baseline.py --graph-dir data/ebird/graph_top100_spatial --epochs 10 --train-positive-edges 1000000 --train-negative-edges 1000000 --eval-positive-edges 500000 --eval-negative-edges 500000
```

Sampled-edge link baseline result:

| Model | Train AUROC | Train AUPRC | Test AUROC | Test AUPRC | Test species macro AUROC | Test species macro AUPRC | Test ECE | Species calibration MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Species embedding + checklist features | 0.8764 | 0.8601 | 0.8761 | 0.8618 | 0.8068 | 0.7125 | 0.0150 | 0.0300 |

Graph link baseline notes:

- This is a non-message-passing bridge model: checklist features plus learned
  species embeddings, trained on sampled positive and negative graph edges.
- The close train/test metrics suggest no obvious overfit under the sampled-edge
  spatial split.
- The AUPRC is evaluated on a balanced sampled-edge test set with 500,000
  positives and 500,000 negatives. It is therefore useful for comparing graph
  link models to one another, but it should not be read as directly comparable
  to the tabular all-species/all-checklist AUPRC values.
- The species macro metrics are lower than the pooled sampled-edge metrics,
  which confirms that species-level difficulty is still important even when the
  aggregate graph link baseline looks strong.
- Aggregate probability-bin calibration is clean: ECE is 0.0150 and max
  probability-bin error is 0.0313.
- The largest remaining calibration errors are species-level. The biggest
  absolute species calibration errors are Fish Crow, Pileated Woodpecker,
  Red-breasted Nuthatch, Eastern Meadowlark, Gray Catbird, Boat-tailed Grackle,
  Mourning Dove, Eastern Phoebe, House Sparrow, and Common Grackle.
- The lowest sampled-edge AUPRC species are Cooper's Hawk, House Sparrow, Green
  Heron, Northern House Wren, Red-breasted Nuthatch, Pine Siskin, Cedar Waxwing,
  Red-tailed Hawk, Red-headed Woodpecker, Eastern Meadowlark, Tree Swallow, and
  Wood Duck.
- Link-baseline runs write
  `species_embedding_link_test_species_metrics.csv` and
  `species_embedding_link_test_calibration.csv` in addition to the aggregate
  metrics JSON.
- All-pairs graph evaluation command:

```
python exp/evaluate_ebird_graph_all_pairs.py --graph-dir data/ebird/graph_top100_spatial
```

- All-pairs graph evaluation scores every held-out checklist crossed with every
  top-100 species, matching the tabular baseline target distribution.

| Graph evaluation target | Micro AUROC | Micro AUPRC | Macro AUROC | Macro AUPRC | ECE | Species calibration MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sampled edges | 0.8761 | 0.8618 | 0.8068 | 0.7125 | 0.0150 | 0.0300 |
| All held-out pairs | 0.8735 | 0.5450 | 0.8230 | 0.3758 | 0.2234 | 0.2234 |
| All held-out pairs, prior corrected | 0.8735 | 0.5450 | 0.8230 | 0.3758 | 0.0089 | 0.0181 |

- The all-pairs evaluation is the fair comparison to the tabular MLP. Under
  that target, the non-message-passing graph link baseline is slightly below the
  MLP ecology + effort baseline on micro AUPRC (0.5450 vs 0.5864) and macro
  AUPRC (0.3758 vs 0.4194), while micro AUROC is also lower (0.8735 vs 0.8921).
- The raw all-pairs calibration error is expected from the current training
  setup: the model was trained on a 50/50 sampled positive/negative edge set,
  then evaluated on the real all-pairs prevalence of 0.1402.
- A global case-control prior correction applies a logit shift of -1.8136,
  moving from train-sample prevalence 0.5000 to all-pairs prevalence 0.1402.
  This reduces ECE from 0.2234 to 0.0089 and species calibration MAE from 0.2234
  to 0.0181. Ranking metrics are unchanged, as expected.
- Species-level graph vs tabular comparison command:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified
```

- The comparison writes
  `top100_mlp_both_spatial-stratified_prior_corrected_all_pairs_graph_vs_tabular_species.csv`
  in the graph link baseline output directory when prior-corrected all-pairs
  metrics are present.
- With all-pairs metrics, graph AUPRC is now comparable to tabular AUPRC. The
  graph link baseline has only tiny AUROC gains for a few species, led by Hooded
  Warbler, Yellow-throated Warbler, Double-crested Cormorant, and House Sparrow.
  Most species are slightly worse than the tabular MLP.
- The largest all-pairs graph AUROC losses vs tabular MLP are Red-headed
  Woodpecker, Eastern Meadowlark, Mourning Dove, Red-winged Blackbird, Great
  Egret, Belted Kingfisher, Yellow-billed Cuckoo, White-throated Sparrow,
  Chipping Sparrow, Red-breasted Nuthatch, Northern House Wren, and Turkey
  Vulture.
- The corrected conclusion is that the species-embedding graph bridge baseline
  is a useful sanity check, and prior correction fixes most of its probability
  calibration, but it does not beat the tabular MLP on the fair all-pairs
  ranking target. The next modeling step should train/evaluate with a loss that
  better reflects the target all-pairs distribution.

Next graph-bridge calibration steps:

1. Apply a global case-control prior correction to the sampled-edge model logits.
   The current link model was trained on a roughly 50/50 positive/negative edge
   sample, while the all-pairs held-out prevalence is 0.1402. A first correction
   is:

   \[
   \operatorname{logit}(p_{\text{corrected}})
   =
   \operatorname{logit}(p_{\text{raw}})
   +
   \operatorname{logit}(\pi_{\text{all-pairs}})
   -
   \operatorname{logit}(\pi_{\text{train-sample}})
   \]

   where \(\pi_{\text{train-sample}}\) is the sampled-edge training prevalence
   and \(\pi_{\text{all-pairs}}\) is the target all-pairs prevalence. This should
   improve calibration without changing AUROC/AUPRC ranking.
2. If global correction fixes most of the ECE but species-level calibration
   remains poor, add species-specific intercept calibration or Platt scaling on
   a calibration split.
3. After that, replace sampled-edge training with a target-aware graph bridge
   objective: mini-batch checklists, score all top-100 species for each
   checklist, and train against the full checklist-by-species label vector. This
   matches the all-pairs evaluation target and should make probabilities more
   meaningful before adding graph message passing.

All-species checklist-batch graph bridge command:

```
python exp/ebird_graph_all_species_baseline.py --graph-dir data/ebird/graph_top100_spatial --architecture pair-mlp --epochs 10 --batch-size 2048 --embedding-dim 32 --hidden-dim 64 --hidden-layers 1 --dropout 0.10
```

This writes outputs under `data/ebird/graph_top100_spatial/all_species_link_baselines`:

- `all_species_link_<architecture>_summary.json`
- `all_species_link_<architecture>_test_species_metrics.csv`
- `all_species_link_<architecture>_test_calibration.csv`
- `all_species_link_<architecture>_history.csv`
- `all_species_link_<architecture>_model.pt`

This bridge model still does not use graph message passing. Its purpose is to
test whether aligning the training objective to the all-pairs target closes the
gap with the tabular MLP before adding GNN complexity.

All-species checklist-batch graph bridge result:

| Model | Micro AUROC | Micro AUPRC | Macro AUROC | Macro AUPRC | ECE | Species calibration MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All-species checklist-batch graph bridge | 0.8809 | 0.5685 | 0.8292 | 0.3944 | 0.0094 | 0.0159 |

Comparison to the best tabular MLP ecology + effort baseline:

| Model | Micro AUROC | Micro AUPRC | Macro AUROC | Macro AUPRC | ECE | Species calibration MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tabular MLP ecology + effort | 0.8921 | 0.5864 | 0.8444 | 0.4194 | 0.0050 | 0.0132 |
| All-species checklist-batch graph bridge | 0.8809 | 0.5685 | 0.8292 | 0.3944 | 0.0094 | 0.0159 |

All-species bridge notes:

- Training on the full checklist-by-species label matrix improves over the
  sampled-edge bridge on the fair all-pairs target: micro AUPRC increases from
  0.5450 to 0.5685 and macro AUPRC increases from 0.3758 to 0.3944.
- Calibration is good without post-hoc prior correction because the training
  objective now matches the all-pairs target prevalence.
- The all-species bridge still does not beat the tabular MLP ecology + effort
  model. It narrows the gap, but the remaining deficit is meaningful: macro
  AUPRC is 0.3944 vs 0.4194 and micro AUPRC is 0.5685 vs 0.5864.
- Species-level comparison shows only small AUROC gains for a few species,
  including Brown Thrasher, Double-crested Cormorant, Brown-headed Nuthatch,
  European Starling, and Hooded Warbler.
- The largest AUROC/AUPRC losses remain Red-headed Woodpecker, Eastern
  Meadowlark, Red-breasted Nuthatch, Tree Swallow, Cedar Waxwing, House Sparrow,
  American Redstart, Blue-headed Vireo, Mallard, Chipping Sparrow, Gray Catbird,
  and Northern House Wren.
- This suggests the bridge architecture itself is now the limiting factor, not
  just the sampled-edge objective. A full GNN should only be added if it
  contributes information that the tabular MLP and checklist/species embedding
  bridge cannot already express, such as spatial/locality/observer graph
  structure or species co-occurrence context.

Bridge architecture experiment:

- `pair-mlp`: the current bridge. It scores each checklist/species pair by
  concatenating checklist features with a learned species embedding and passing
  that pair through an MLP.
- `factorized`: encodes each checklist once, then scores species with a
  low-rank checklist-latent by species-embedding dot product plus checklist and
  species biases. This is more explicitly matrix-factorized and efficient, but
  may underfit species-specific nonlinear responses.
- `hybrid`: adds a direct multi-species checklist head to the factorized dot
  product. This is closest to the tabular MLP while retaining an explicit
  species-embedding interaction term. This became the main bridge architecture
  for the later spatial residual and spatial-cell GNN tests.

Recommended bridge architecture commands:

```
python exp/ebird_graph_all_species_baseline.py --graph-dir data/ebird/graph_top100_spatial --architecture hybrid --epochs 10 --batch-size 2048 --hidden-dim 64 --hidden-layers 1 --latent-dim 64 --dropout 0.10
python exp/ebird_graph_all_species_baseline.py --graph-dir data/ebird/graph_top100_spatial --architecture factorized --epochs 10 --batch-size 2048 --hidden-dim 64 --hidden-layers 1 --latent-dim 64 --dropout 0.10
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/all_species_link_baselines/all_species_link_hybrid_test_species_metrics.csv
```

Bridge architecture results:

| Model | Micro AUROC | Micro AUPRC | Macro AUROC | Macro AUPRC | ECE | Species calibration MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tabular MLP ecology + effort | 0.8921 | 0.5864 | 0.8444 | 0.4194 | 0.0050 | 0.0132 |
| Pair MLP all-species bridge | 0.8809 | 0.5685 | 0.8292 | 0.3944 | 0.0094 | 0.0159 |
| Factorized all-species bridge | 0.8859 | 0.5768 | 0.8364 | 0.4029 | 0.0140 | 0.0186 |
| Hybrid all-species bridge | 0.8899 | 0.5840 | 0.8417 | 0.4117 | 0.0076 | 0.0144 |
| Stronger hybrid all-species bridge | 0.8935 | 0.5895 | 0.8476 | 0.4242 | 0.0038 | 0.0136 |
| Stronger hybrid + locality/spatial prior | 0.8605 | 0.5143 | 0.8059 | 0.3455 | 0.0130 | 0.0207 |
| Stronger hybrid + locality/spatial scalars | 0.8671 | 0.5325 | 0.8217 | 0.3790 | 0.0181 | 0.0315 |
| Stronger hybrid + locality/spatial prior, weight init 0 | 0.8694 | 0.5313 | 0.8209 | 0.3725 | 0.0195 | 0.0252 |
| Stronger hybrid + spatial-neighbor scalars | 0.8859 | 0.5692 | 0.8406 | 0.4119 | 0.0057 | 0.0234 |
| Stronger hybrid + spatial-neighbor prior, weight init 0 | 0.8859 | 0.5710 | 0.8402 | 0.4136 | 0.0063 | 0.0231 |
| Stronger hybrid + RBF spatial residual | 0.8933 | 0.5910 | 0.8458 | 0.4248 | 0.0108 | 0.0144 |
| Spatial-cell GCN hybrid | 0.8910 | 0.5828 | 0.8455 | 0.4182 | 0.0109 | 0.0195 |
| Spatial-cell GCN residual hybrid | 0.8923 | 0.5857 | 0.8476 | 0.4280 | 0.0136 | 0.0215 |
| Spatial-cell GCN gated residual hybrid | 0.8922 | 0.5877 | 0.8478 | 0.4258 | 0.0153 | 0.0218 |
| Spatial-cell GCN residual, 64 hidden, 1 layer, wd 1e-4 | 0.8944 | 0.5927 | 0.8484 | 0.4287 | 0.0085 | 0.0144 |
| Spatial-cell GCN residual, 64 hidden, 1 layer, wd 1e-3 | 0.8944 | 0.5928 | 0.8484 | 0.4284 | 0.0087 | 0.0146 |

Hybrid bridge notes:

- The hybrid architecture is now very close to the tabular MLP. It closes most
  of the gap left by the pair MLP bridge: micro AUPRC improves from 0.5685 to
  0.5840, and macro AUPRC improves from 0.3944 to 0.4117.
- The factorized model improves over the pair MLP but trails the hybrid model,
  suggesting the direct multi-species checklist head is carrying useful
  checklist-to-species signal beyond the low-rank species interaction.
- The hybrid bridge still narrowly trails the tabular MLP on all aggregate
  ranking metrics: micro AUPRC 0.5840 vs 0.5864 and macro AUPRC 0.4117 vs
  0.4194. Calibration is close, with ECE 0.0076.
- The stronger hybrid capacity check (`hidden-dim=128`, `hidden-layers=2`,
  `latent-dim=128`) now slightly beats the tabular MLP on all aggregate ranking
  metrics: micro AUPRC 0.5895 vs 0.5864, macro AUPRC 0.4242 vs 0.4194, micro
  AUROC 0.8935 vs 0.8921, and macro AUROC 0.8476 vs 0.8444. It also improves
  ECE to 0.0038. Species calibration MAE is slightly worse than the tabular MLP
  at 0.0136 vs 0.0132, but the gap is small.
- Hybrid species-level gains over tabular are now real for some species. The
  stronger hybrid's largest AUPRC gains include Bald Eagle, Brown Thrasher, Gray
  Catbird, Black-and-white Warbler, Osprey, Pied-billed Grebe, Summer Tanager,
  Yellow-throated Warbler, Eastern Towhee, Canada Goose, Indigo Bunting, and
  Ruby-throated Hummingbird.
- The largest remaining stronger-hybrid losses include Red-headed Woodpecker,
  Wood Duck, Boat-tailed Grackle, Swamp Sparrow, Great Black-backed Gull,
  Bufflehead, Purple Martin, American Herring Gull, Brown-headed Cowbird,
  Chipping Sparrow, Mourning Dove, and Brown Pelican.
- Species calibration is also now comparable across the tabular and bridge
  outputs. The tabular MLP species calibration MAE is 0.0132, slightly better
  than the stronger hybrid bridge at 0.0136.
- The stronger hybrid result means the previous deficit was mostly architecture
  capacity, not evidence that graph message passing is required. A full GNN
  should only be added after a focused relational-feature baseline, and it
  should add information not present in the current feature matrix:
  locality/hotspot repeated-visit structure, spatial neighbor edges, observer
  effects, or checklist co-detection context.
- The first locality/spatial prior run was a clear regression. It reduced micro
  AUPRC from 0.5895 to 0.5143 and macro AUPRC from 0.4242 to 0.3455 relative to
  the stronger hybrid, with species calibration MAE worsening from 0.0136 to
  0.0207. The learned prior-logit weight increased to about 1.71, so the model
  appears to have overused train-only locality/grid priors that do not transfer
  cleanly under the spatial-block holdout.
- This failed run is still useful: it argues against injecting same-locality or
  same-grid species priors as a strong additive shortcut. The next relational
  baseline should separate transferable scalar locality/spatial information
  from direct species prior logits.
- The follow-up locality/spatial scalar-only run also regressed: micro AUPRC was
  0.5325 and macro AUPRC was 0.3790, with species calibration MAE worsening to
  0.0315. This means the four simple locality/grid aggregate scalar features
  are not helping the stronger hybrid transfer under the current spatial-block
  split.
- Retrying the per-species locality/spatial prior with prior-logit weight
  initialized at zero was also worse than the plain stronger hybrid: micro AUPRC
  0.5313, macro AUPRC 0.3725, ECE 0.0195, and species calibration MAE 0.0252.
  This is better than the original weight-1 prior run on some species, but still
  not close to the plain stronger hybrid.
- Current conclusion: do not carry the locality/grid aggregate feature design
  forward as the main relational path. It likely over-emphasizes sparse local
  history or sampling geography that does not generalize across held-out spatial
  blocks. The next relational test should use smoother transfer mechanisms,
  such as distance-weighted neighbor summaries from nearby training cells,
  spatial graph edges, or locality/month repeated-visit structure with stronger
  regularization.
- The spatial-neighbor scalar baseline is much healthier than the failed
  same-cell/locality aggregates, but it still trails the plain stronger hybrid:
  micro AUPRC 0.5692 vs 0.5895 and macro AUPRC 0.4119 vs 0.4242. It also has
  worse species calibration MAE, 0.0234 vs 0.0136. This suggests that smoother
  spatial summaries are useful for some species but are not yet a net improvement
  over the checklist-only stronger hybrid.
- Species-level gains from spatial-neighbor scalars are concentrated in several
  water-associated or spatially clustered species, including Bufflehead,
  Double-crested Cormorant, Pied-billed Grebe, Canada Goose, Mallard, Bald
  Eagle, Great Blue Heron, and Tree Swallow. Losses remain for species such as
  Gray Catbird, House Finch, Pileated Woodpecker, Wood Duck, Yellow-billed
  Cuckoo, Ovenbird, Scarlet Tanager, Red-eyed Vireo, Field Sparrow, and Wood
  Thrush.
- The current spatial-neighbor scalar result does not justify replacing the
  stronger hybrid as the default baseline. It does justify one controlled retry
  with smoothed per-species spatial-neighbor prior logits initialized at zero,
  because the scalar-only version improved selected spatially clustered species
  but may be too compressed to express species-specific spatial structure.
- The spatial-neighbor prior-logit retry improved slightly over the scalar-only
  spatial-neighbor run: micro AUPRC 0.5710 vs 0.5692, macro AUPRC 0.4136 vs
  0.4119, and species calibration MAE 0.0231 vs 0.0234. The learned prior-logit
  weight stayed modest at about 0.21, unlike the failed locality prior run.
- Even with the prior logits, spatial-neighbor augmentation still trails the
  plain stronger hybrid: micro AUPRC 0.5710 vs 0.5895 and macro AUPRC 0.4136 vs
  0.4242. The next step should not be more aggregate-feature variants unless
  there is a specific ecological hypothesis. Move to a real graph/message-
  passing baseline or an explicit spatial residual formulation.
- Species-level gains from the spatial-neighbor prior are again concentrated in
  water-associated or spatially clustered species: Bufflehead, Pied-billed
  Grebe, Double-crested Cormorant, Canada Goose, Mallard, Great Blue Heron, Bald
  Eagle, Eastern Meadowlark, American Robin, Tree Swallow, Hooded Merganser, and
  Great Egret. The largest remaining losses include Great Black-backed Gull,
  Wood Duck, Gray Catbird, American Herring Gull, Field Sparrow, Pileated
  Woodpecker, House Finch, Red-headed Woodpecker, Yellow-billed Cuckoo,
  Ovenbird, Red-eyed Vireo, and Brown Thrasher.
- Before the first successful spatial-cell GNN grid, the explicit RBF spatial
  residual was the best aggregate ranking model:
  micro AUPRC 0.5910 and macro AUPRC 0.4248. It improves slightly over the plain
  stronger hybrid on micro AUPRC (0.5910 vs 0.5895) and essentially ties/slightly
  improves macro AUPRC (0.4248 vs 0.4242), while keeping micro AUROC very close
  (0.8933 vs 0.8935).
- The spatial residual's tradeoff is calibration. Probability-bin ECE worsens
  from 0.0038 for the plain stronger hybrid to 0.0108, and species calibration
  MAE worsens from 0.0136 to 0.0144. This is still not a severe calibration
  failure, but it matters because bias/effort modeling needs probability
  estimates, not only ranking.
- Species-level gains from the spatial residual are broader than the
  spatial-neighbor aggregate gains. The largest AUPRC improvements over the
  tabular MLP include Pied-billed Grebe, Black-and-white Warbler,
  Double-crested Cormorant, Bufflehead, Bald Eagle, Brown Thrasher,
  Yellow-throated Warbler, American Redstart, Great Egret, White-eyed Vireo,
  Canada Goose, and Gray Catbird.
- The largest AUPRC losses for the spatial residual include Red-headed
  Woodpecker, Wood Duck, Northern Rough-winged Swallow, Swamp Sparrow, Mallard,
  Great Black-backed Gull, Boat-tailed Grackle, American Herring Gull, Purple
  Martin, Red-winged Blackbird, Brown-headed Cowbird, and Chipping Sparrow.
- Current conclusion: the spatial residual is a useful final non-GNN benchmark.
  It proves that smooth leftover spatial structure can improve ranking, but the
  calibration penalty reinforces the original concern: spatial structure can
  also encode observer geography or clustered sampling bias. A GNN should now be
  tested against both the plain stronger hybrid and the spatial residual model.
- The first true message-passing GNN, a spatial-cell GCN feeding the hybrid
  detection head, does not beat the stronger hybrid or the RBF spatial residual.
  It achieves micro AUPRC 0.5828 and macro AUPRC 0.4182, below the stronger
  hybrid's 0.5895/0.4242 and the spatial residual's 0.5910/0.4248. Its
  calibration is also weaker: ECE 0.0109 and species calibration MAE 0.0195.
- The spatial-cell GCN still improves some spatially clustered or effort-biased
  species relative to the tabular MLP, including Double-crested Cormorant,
  Eastern Meadowlark, Mallard, Black-and-white Warbler, Bald Eagle, American
  Robin, Indigo Bunting, Canada Goose, Great Blue Heron, Eastern Towhee,
  Yellow-throated Warbler, and Hooded Warbler. Losses include Red-headed
  Woodpecker, Swamp Sparrow, Brown Pelican, Gray Catbird, Ovenbird, Acadian
  Flycatcher, Great Egret, Great Black-backed Gull, Wood Duck, American Herring
  Gull, Pileated Woodpecker, and Bufflehead.
- Current GNN conclusion: message passing is not automatically helpful. The
  first GCN likely smooths spatial context too generically and loses some of the
  stronger hybrid's checklist-level signal. The next GNN iteration should be
  residual/gated rather than replacing checklist context with cell context:
  start from the stronger hybrid logits and add a small learned GNN residual, or
  use a learned gate that can ignore spatial-cell messages when they hurt
  calibration or species transfer.

Completed modeling sequence:

1. Completed: add species-level calibration to the tabular MLP metrics so tabular and graph
   bridge outputs can be compared on the same species calibration diagnostics.
   The tabular metrics should include per-species `mean_predicted` and
   `calibration_error`, with `species_calibration_mae` in the summary JSON.
2. Completed: run one stronger hybrid bridge as a capacity check:

```
python exp/ebird_graph_all_species_baseline.py --graph-dir data/ebird/graph_top100_spatial --architecture hybrid --run-name hybrid_h128_l2_z128 --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --dropout 0.10
```

3. The stronger hybrid slightly beats the tabular MLP on aggregate ranking
   metrics. Treat this as evidence that bridge capacity, not message passing,
   explained most of the previous gap.
4. Completed: build locality/spatial-neighbor enriched bridges before adding
   message passing. These tested whether relational information helps when added
   as train-only aggregate features.
5. Prioritize locality/spatial structure before observer structure. Observer
   effects are likely strong but can dominate ecology and should be introduced
   carefully after cleaner spatial/locality signals are tested.
6. Candidate relational features for the enriched bridge:

   - train-only locality or hotspot visit count
   - train-only locality species detection rates
   - spatial-neighborhood checklist density
   - spatial-neighborhood species detection rates
   - repeated-visit count by locality/month or locality/season
   - local species richness or co-detection summaries fit from training data

Locality/spatial enriched bridge implementation:

- `exp/ebird_graph_all_species_baseline.py` supports
  `--feature-augmentation locality-spatial` and
  `--feature-augmentation locality-spatial-scalars`.
- The augmentation adds train-only scalar checklist features:

  - `locality_train_checklists_log1p`
  - `spatial_cell_train_checklists_log1p`
  - `locality_train_species_rate_mean`
  - `spatial_cell_train_species_rate_mean`

- It also adds a per-checklist/species prior logit matrix based on smoothed
  train-only locality and spatial-cell species detection rates. Training rows
  use leave-one-out rates so the checklist's own labels are not copied into its
  features; held-out rows use only training split aggregates.
- `locality-spatial-scalars` uses only the four scalar checklist features and
  omits the per-species prior logits. This is the next diagnostic run because
  it tests whether repeated-visit/checklist-density information helps without
  directly encoding species-specific hotspot priors.
- For `locality-spatial`, `--prior-logit-weight` now defaults to 0.0. The prior
  logit weight remains learnable, but starting from zero avoids injecting the
  train-only prior as a strong shortcut before the model proves it helps.
- Spatial cells default to 25 km in the analysis CRS. This is intentionally a
  simple pre-GNN relational feature test rather than message passing.

Locality/spatial prior run that regressed:

```
python exp/ebird_graph_all_species_baseline.py --graph-dir data/ebird/graph_top100_spatial --architecture hybrid --feature-augmentation locality-spatial --run-name hybrid_h128_l2_z128_locality_spatial --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --dropout 0.10 --spatial-grid-size-m 25000 --prior-smoothing 20
```

Locality/scalar diagnostic that also regressed:

```
python exp/ebird_graph_all_species_baseline.py --graph-dir data/ebird/graph_top100_spatial --architecture hybrid --feature-augmentation locality-spatial-scalars --run-name hybrid_h128_l2_z128_locality_spatial_scalars --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --dropout 0.10 --spatial-grid-size-m 25000 --prior-smoothing 20
```

Prior-logit retry initialized at zero that also regressed:

```
python exp/ebird_graph_all_species_baseline.py --graph-dir data/ebird/graph_top100_spatial --architecture hybrid --feature-augmentation locality-spatial --run-name hybrid_h128_l2_z128_locality_spatial_w0 --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --dropout 0.10 --spatial-grid-size-m 25000 --prior-smoothing 20 --prior-logit-weight 0
```

Comparison command:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/all_species_link_baselines/all_species_link_hybrid_h128_l2_z128_locality_spatial_scalars_test_species_metrics.csv
```

Completed relational/GNN modeling sequence:

1. Keep the plain stronger hybrid as the current best bridge baseline.
2. Completed: add a smoother spatial-neighbor relational baseline rather than
   same-cell or same-locality aggregate priors. The implemented version
   aggregates training checklists to grid cells, then builds distance-weighted
   summaries over nearby training grid-cell centroids with a minimum-cell
   threshold and shrinkage toward global prevalence.
3. Completed: run the spatial-neighbor prior-logit diagnostic initialized at
   zero. It modestly improves over the spatial-neighbor scalar run but still
   trails the plain stronger hybrid.
4. Completed: fit an explicit smooth spatial residual bridge before the first
   full message-passing GNN. This gives a stronger and more interpretable
   non-GNN benchmark: ecology, effort, temporal covariates, species
   interactions, plus a constrained spatial correction term.
5. Completed: build the first true message-passing GNN and compare it against
   both the plain stronger hybrid and the RBF spatial residual. The GNN needs to
   improve ranking without worsening calibration enough to undermine the
   bias/effort modeling goal.
6. Completed: first spatial-cell GCN hybrid. It underperforms the stronger
   hybrid and RBF spatial residual, so the next GNN should be residual/gated
   instead of directly concatenating cell message-passing context into the
   checklist encoder.
7. Completed: run residual and gated spatial-cell GCNs. These keep the stronger
   checklist/species hybrid as the base detector and add spatial-cell message
   passing only as a learned correction. This tests whether graph structure can
   improve spatially clustered or effort-biased species without degrading the
   strong checklist-level signal and calibration.
8. Completed: diagnose the residual/gated GNN tradeoff before adding more graph
   complexity. The residual/gated GNNs improved over the direct concat GCN, but
   the first untuned versions did not beat the RBF spatial residual on micro
   AUPRC and worsened calibration. This led to the residual grid search.
9. Current: favor the one-layer, 64-hidden residual spatial-cell GCN as the best
   GNN candidate so far. It improves aggregate ranking over the RBF spatial
   residual and keeps calibration better than the RBF residual, but species-level
   failures remain, especially Red-headed Woodpecker.

First spatial-cell GNN:

- `exp/ebird_spatial_gnn_baseline.py` implements the first conservative
  message-passing baseline.
- It keeps the all-species checklist-batch objective and the hybrid
  checklist/species detection head, but adds a spatial-cell GCN before
  detection scoring:

  1. assign each checklist to a 25 km spatial grid cell
  2. create spatial-cell nodes
  3. build queen-neighbor spatial-cell edges
  4. initialize cell features from train-only checklist summaries plus cell
     coordinates and train checklist counts
  5. run GCN message passing across spatial cells
  6. concatenate each checklist's original features with its spatial-cell GNN
     embedding
  7. score all top-100 species with the hybrid all-species detection head

- This is the first actual GNN test in the workflow. It asks whether learned
  spatial message passing improves over the stronger hybrid and over the RBF
  spatial residual without relying on pseudo-absence background sampling.
- The main comparison target is still the all held-out checklist/species pairs
  under the same spatial-stratified split.

Recommended first spatial GNN command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial --run-name spatial_gcn_h128_l2_z128_cell64_l2 --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 2 --dropout 0.10 --spatial-grid-size-m 25000
```

Comparison command:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/spatial_gnn_baselines/spatial_gnn_spatial_gcn_h128_l2_z128_cell64_l2_test_species_metrics.csv
```

First spatial GNN result:

- Micro AUROC 0.8910, micro AUPRC 0.5828.
- Macro AUROC 0.8455, macro AUPRC 0.4182.
- ECE 0.0109, max probability-bin error 0.0496.
- Species calibration MAE 0.0195.
- Graph: 268 spatial-cell nodes, 2,164 spatial-cell edges, 262 cells with
  training checklists.
- This is below the stronger hybrid and below the RBF spatial residual. The next
  GNN should use message passing as an additive/gated residual over the stronger
  hybrid, not as a direct replacement for the bridge context.

Residual/gated spatial-cell GNN:

- `exp/ebird_spatial_gnn_baseline.py` now supports `--gnn-mode concat`,
  `--gnn-mode residual`, and `--gnn-mode gated`.
- `concat` is the original spatial-cell GCN: concatenate each checklist's cell
  embedding into the checklist encoder.
- `residual` keeps the stronger checklist-only hybrid path and adds a
  zero-initialized species-specific residual from the spatial-cell GCN:

  \[
  \operatorname{logit} P(y_{c,j}=1)
  =
  f_{\text{hybrid}}(x_c, j)
  +
  r_j(g_{\text{cell}}(s_c))
  \]

- `gated` uses the same residual but multiplies it by a learned
  checklist/species gate. The gate starts conservative, so the model can ignore
  spatial-cell messages when they hurt transfer or calibration.
- These modes are closer to the overall bias/effort goal than the direct concat
  GCN because they treat spatial graph structure as a correction to
  ecology/effort/species interactions, not as a replacement for those
  covariates. A useful GNN should beat or tie the stronger hybrid and RBF
  residual on ranking while preserving calibration.

Residual/gated spatial GNN results:

- Residual GCN: micro AUROC 0.8923, micro AUPRC 0.5857, macro AUROC 0.8476,
  macro AUPRC 0.4280, ECE 0.0136, species calibration MAE 0.0215.
- Gated residual GCN: micro AUROC 0.8922, micro AUPRC 0.5877, macro AUROC
  0.8478, macro AUPRC 0.4258, ECE 0.0153, species calibration MAE 0.0218.
- The first residual/gated design improves over the direct concat GCN,
  especially for macro AUPRC. These untuned residual/gated runs were later
  superseded by the one-layer residual grid results.
- These are still not decisive wins over the non-GNN benchmarks. The gated run
  approaches the stronger hybrid's micro AUPRC (0.5877 vs 0.5895), and the
  residual run exceeds the stronger hybrid/RBF residual on macro AUPRC (0.4280
  vs 0.4242/0.4248), but both GNN variants have weaker calibration than the
  stronger hybrid and RBF residual.
- Species-level residual gains are concentrated in species where spatial
  context plausibly helps: Black-and-white Warbler, Mallard, Eastern
  Meadowlark, Double-crested Cormorant, Indigo Bunting, Yellow-billed Cuckoo,
  Great Blue Heron, Yellow-throated Warbler, Bald Eagle, Hooded Warbler,
  Hooded Merganser, and Blue Grosbeak. Losses still include Red-headed
  Woodpecker, Swamp Sparrow, Wood Duck, Brown Pelican, Pileated Woodpecker,
  Great Black-backed Gull, Green Heron, Red-winged Blackbird, Common Grackle,
  Ovenbird, Wood Thrush, and Boat-tailed Grackle.
- The gated comparison is similar but slightly more polarized. Its largest
  AUPRC gains over the tabular MLP are Eastern Meadowlark (+0.1171),
  Black-and-white Warbler (+0.0993), Mallard (+0.0956), Double-crested
  Cormorant (+0.0621), American Robin (+0.0493), Bald Eagle (+0.0483),
  Indigo Bunting (+0.0445), Great Blue Heron (+0.0410), Hooded Warbler
  (+0.0380), Hooded Merganser (+0.0353), Eastern Towhee (+0.0352), and
  Canada Goose (+0.0303).
- The gated model's largest AUPRC losses are Red-headed Woodpecker (-0.1224),
  Boat-tailed Grackle (-0.0558), Swamp Sparrow (-0.0380), Green Heron
  (-0.0322), Acadian Flycatcher (-0.0309), Wood Duck (-0.0302), Ovenbird
  (-0.0253), Barn Swallow (-0.0245), Wood Thrush (-0.0239), Brown Pelican
  (-0.0212), Yellow-billed Cuckoo (-0.0197), and American Herring Gull
  (-0.0191). The large Red-headed Woodpecker loss is a warning that the gate is
  not yet reliably protecting species where spatial smoothing is harmful.
- The gated model's largest species calibration errors are Pileated Woodpecker
  and American Crow (both 0.0785), followed by Mourning Dove (0.0662),
  White-breasted Nuthatch (0.0613), Turkey Vulture (0.0595), Pine Warbler
  (0.0546), Chipping Sparrow (0.0500), and Double-crested Cormorant (0.0478).
  Calibration is therefore the main weakness of the current residual/gated
  spatial-cell GNN family.
- Interim interpretation before the grid search: residual/gated message passing
  was the right direction relative to concat, but the GNN was still acting like
  an imperfect spatial correction. That motivated tuning residual strength and
  spatial-cell capacity before adding graph nodes for locality, protocol/effort,
  and environmental neighborhoods.
- The next tuning step should be a small, resumable grid over residual/gated
  spatial-cell GNN capacity and regularization. The goal is not to maximize one
  ranking metric at any cost; it is to find whether graph message passing can
  improve species-level ranking while keeping calibration close to the stronger
  hybrid and RBF spatial residual baselines.
- The first residual-only grid pass is encouraging. The best runs use a single
  spatial-cell GCN layer with 64 hidden units. They beat the RBF spatial
  residual on ranking while keeping calibration better than the RBF residual:

  - `spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001`: micro AUPRC
    0.5927, macro AUPRC 0.4287, ECE 0.0085, species calibration MAE 0.0144.
  - `spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p001`: micro AUPRC
    0.5928, macro AUPRC 0.4284, ECE 0.0087, species calibration MAE 0.0146.

- This is the first GNN result that is clearly competitive with the non-GNN
  spatial residual. It improves micro AUPRC over the RBF residual (0.5928 vs
  0.5910), improves macro AUPRC (0.4284/0.4287 vs 0.4248), and improves ECE
  relative to the RBF residual (0.0085-0.0087 vs 0.0108). It still does not
  match the plain stronger hybrid's ECE (0.0038), so calibration remains the
  main constraint.
- Two-layer residual GCNs are worse than one-layer residual GCNs in this grid,
  especially with 64 hidden units. This suggests over-smoothing or excessive
  spatial correction, not under-capacity.
- Species-level diagnostics for the best residual grid run
  (`spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001`) show that the
  aggregate improvement is ecologically plausible but not uniformly beneficial.
  Largest AUPRC gains over the tabular MLP are Black-and-white Warbler
  (+0.1119), Double-crested Cormorant (+0.0877), Mallard (+0.0656),
  Yellow-billed Cuckoo (+0.0637), Eastern Meadowlark (+0.0459), American Robin
  (+0.0442), Yellow-throated Warbler (+0.0438), Pied-billed Grebe (+0.0425),
  Indigo Bunting (+0.0421), Great Blue Heron (+0.0401), Great Egret (+0.0368),
  and Brown Thrasher (+0.0367).
- Largest AUPRC losses for that same run are Red-headed Woodpecker (-0.1067),
  Swamp Sparrow (-0.0581), Wood Duck (-0.0480), Great Black-backed Gull
  (-0.0365), Pileated Woodpecker (-0.0285), American Herring Gull (-0.0197),
  Ovenbird (-0.0191), Green Heron (-0.0174), Brown-headed Cowbird (-0.0153),
  Royal Tern (-0.0148), Red-winged Blackbird (-0.0126), and Bufflehead
  (-0.0117). Red-headed Woodpecker remains the clearest failure case and should
  be inspected before claiming the GNN is broadly better.
- The largest species calibration errors for the best residual run are
  White-breasted Nuthatch (0.0600), Chipping Sparrow (0.0542), Mourning Dove
  (0.0366), American Goldfinch (0.0319), European Starling (0.0317), Hairy
  Woodpecker (0.0304), Pileated Woodpecker (0.0295), Northern Cardinal (0.0287),
  Turkey Vulture (0.0287), Mallard (0.0286), Ruby-crowned Kinglet (0.0272), and
  Pine Warbler (0.0270). These are lower than the earlier gated model's worst
  calibration failures, which supports favoring the residual 64-hidden,
  one-layer configuration for now.
- The completed 24-run residual/gated grid confirms that conclusion rather than
  overturning it. The best aggregate ranking run remains the one-layer,
  64-hidden residual GCN with weight decay 1e-3:
  `spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p001` has micro AUPRC
  0.5928, macro AUPRC 0.4284, ECE 0.0087, and species calibration MAE 0.0146.
  The nearly tied weight-decay 1e-4 run has slightly lower micro AUPRC but
  slightly better macro AUPRC, ECE, and species calibration MAE, so it remains
  the preferred primary model for interpretation.
- The best gated GCNs are competitive but do not clearly surpass the residual
  models. The strongest calibration/ranking compromise is
  `spatial_gcn_gated_h128_l2_z128_cell64_cl1_wd0p0001_gbm2` with micro AUPRC
  0.5923, macro AUPRC 0.4282, ECE 0.0078, and species calibration MAE 0.0144.
  This makes it a useful sensitivity run, not the main model.
- Gating is most useful when it is shallow and moderately permissive. The
  gate-bias -2 runs generally beat or tie the gate-bias -3 runs on ranking.
  The stricter -3 initialization can improve ECE in some cases, but often gives
  up AUPRC.
- The grid gives a strong warning against adding spatial-cell GCN depth before
  solving diagnostics. Two-layer 64-hidden gated GCNs are among the weakest
  spatial GNN runs, with micro AUPRC around 0.5875-0.5881 and species
  calibration MAE around 0.021-0.022. This is consistent with over-smoothing or
  excessive spatial correction.
- Current interpretation: the spatial-cell GNN is now doing something useful,
  but its gain is still a small spatial residual improvement over a strong
  tabular/bridge baseline. The next step should be diagnostic and validation
  work, not a larger architecture. The main question is whether the GNN is
  improving transferable species-environment/effort structure or just adding a
  smoother version of observer geography.

Next steps after the 24-run grid:

1. Treat `spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001` as the primary
   spatial GNN candidate and
   `spatial_gcn_gated_h128_l2_z128_cell64_cl1_wd0p0001_gbm2` as the calibration
   sensitivity candidate.
2. Run species-level comparisons for the gated sensitivity model and compare
   the species gain/loss list against the primary residual model. If the gated
   run reduces the worst residual species losses without losing much aggregate
   ranking, keep both in the report.
3. Diagnose the repeated failure species, especially Red-headed Woodpecker,
   Swamp Sparrow, Wood Duck, Green Heron, and Ovenbird. For each, inspect
   spatial distribution of positives, held-out block coverage, prevalence,
   effort strata, and whether the GNN correction is pushing probabilities in
   ecologically plausible directions.
4. Add a spatial correction visualization for one or two representative species:
   one species with a large gain and one with a large loss. The plot should show
   the GNN residual or probability difference relative to the stronger hybrid,
   not just final predicted probability.
5. Before adding richer heterogeneous graph nodes, repeat the best residual
   candidate under at least one alternate spatial split seed or spatial-block
   granularity. A real GNN gain should be stable under a changed held-out
   geography.
6. Only after those diagnostics should the graph be expanded to include richer
   relation types, such as species co-detection edges, environmental-neighbor
   cell edges, protocol/effort context nodes, locality nodes, or observer nodes.

Primary-vs-gated species diagnostics:

- The graph-vs-tabular comparison now uses short, stable output filenames to
  avoid Windows path-length failures:
  `residual_primary_graph_vs_tabular_species.csv` and
  `gated_gbm2_graph_vs_tabular_species.csv`.
- The gated sensitivity model is not a replacement for the residual primary
  model. It improves some species more strongly, especially Eastern Meadowlark
  (+0.0926 AUPRC vs +0.0459 for residual), Mallard (+0.0778 vs +0.0656),
  American Robin (+0.0595 vs +0.0442), and Great Egret (+0.0507 vs +0.0368).
  But it worsens some key failure species: Red-headed Woodpecker (-0.1140 vs
  -0.1067), Wood Duck (-0.0666 vs -0.0480), Bufflehead (-0.0576 vs -0.0117),
  and Great Black-backed Gull (-0.0412 vs -0.0365).
- Gating can reduce calibration error for some failure species while still
  hurting ranking. Red-headed Woodpecker's gated calibration error is low
  (0.0038), but its AUPRC loss is worse than the residual model. This means
  species-level calibration alone is not enough; spatial ranking diagnostics are
  still needed.
- The repeated failure species are not simply too sparse. In the held-out split,
  Red-headed Woodpecker has 8,150 positive checklists across 25 positive
  spatial cells and 17 counties; Swamp Sparrow has 7,812 positives across 26
  cells and 20 counties; Wood Duck has 5,382 positives across 24 cells and 19
  counties. These are adequate for diagnosis, so the failure is more likely a
  spatial/effort interaction or an over-smoothing problem.
- Some coastal or localized species have narrower held-out spatial support and
  should be interpreted separately. Boat-tailed Grackle appears in only one
  held-out spatial block and 9 positive cells; Great Black-backed Gull appears
  in two held-out blocks and 8 positive cells. For these species, split geometry
  can dominate model differences.
- Effort/protocol structure is strong for both gains and losses. For example,
  Red-headed Woodpecker prevalence is 0.0916 on traveling held-out checklists
  versus 0.0157 on stationary checklists; Swamp Sparrow is 0.0928 versus
  0.0075; Wood Duck is 0.0618 versus 0.0083; Black-and-white Warbler is 0.0700
  versus 0.0157. The next diagnostic should therefore inspect whether spatial
  residuals are learning species geography or implicitly amplifying the
  traveling-checklist effort pattern.
- Current conclusion after the first diagnostics: keep the residual primary
  model as the main spatial GNN result, keep the gated model as a sensitivity
  check, and move next to residual/probability-difference maps for one large
  gain species and one large loss species.
- The first residual probability-difference maps were generated for
  Black-and-white Warbler, Eastern Meadowlark, Red-headed Woodpecker, Swamp
  Sparrow, Wood Duck, and Green Heron. These maps compare the trained spatial
  GNN's full prediction to its own base checklist/species path before adding the
  spatial residual; they do not directly show the full graph-vs-tabular
  difference.
- The primary residual GNN mostly applies a negative probability correction for
  these focus species. This is especially strong for failure species:
  Red-headed Woodpecker mean probability changes from 0.1268 to 0.0699, with
  positive-checklist mean delta -0.0887; Green Heron changes from 0.0580 to
  0.0334, with positive-checklist mean delta -0.0862; Swamp Sparrow positives
  have mean delta -0.0430. This supports the hypothesis that some failures come
  from overly aggressive spatial down-correction in held-out areas.
- The gain species are more subtle. Black-and-white Warbler and Eastern
  Meadowlark also receive negative average residual corrections, but their
  AUPRC improves against the tabular baseline. This means the residual can still
  improve ranking if it downweights negatives or low-quality effort/geography
  contexts more than it harms the highest-ranked positives. These species should
  be inspected on the map rather than interpreted from mean deltas alone.
- All diagnostic maps now draw the NC state boundary from
  `data/boundaries/nc_state_boundary.gpkg`, projected to EPSG:5070 by default.
  This makes coastal/island and edge behavior easier to distinguish from empty
  plot area.

How to interpret the residual/probability-difference maps:

- Read the residual map as "what the spatial GNN changed after the
  checklist/species base model." Red means the spatial residual increased the
  detection probability; blue means it decreased the detection probability.
  This is not the final prediction surface and it is not a direct
  graph-vs-tabular map.
- Compare the residual panel to the held-out positive panel. A useful residual
  should generally increase probabilities where held-out positives cluster or
  decrease probabilities in areas with many held-out checklists but few
  positives. A concerning residual suppresses broad areas that contain many
  positives or boosts areas where positives are absent.
- Look for whether the residual correction follows plausible species geography
  or simply follows observer geography. A correction aligned with habitat or
  range structure is more encouraging than one aligned only with high-effort
  corridors, hotspots, or coastal/island checklist density.
- Look for edge and coastal artifacts. The NC boundary helps distinguish real
  coastal/island structure from empty plot space. If strong red/blue correction
  occurs only at the state edge, coastline, or isolated islands, treat it as a
  possible split/coverage artifact.
- Look for one-sided corrections. In the current primary residual GNN, several
  focus species receive mostly negative corrections. That is not automatically
  wrong, but it is suspicious for species where held-out positives are also
  being suppressed.
- Interpret gains and losses through ranking, not just average residual. A
  species can have a negative mean residual and still gain AUPRC if the residual
  downweights negatives more than it downweights the highest-ranked positives.
  This is why the delta map and the positive-location map need to be read
  together.

Effort-stratified graph-vs-tabular diagnostics:

- `exp/compare_ebird_effort_strata.py` retrains the tabular MLP on the graph
  dataset's standardized checklist features and compares it against the saved
  spatial GNN on the same held-out all-pairs target within effort and spatial
  strata.
- The corrected first run used the original all-checklist spatial block
  assignment, not a recomputed grid over only held-out rows.
- The spatial GNN's biggest micro-AUPRC gains occur in higher-effort contexts:
  distance `(2,5]` km (+0.0134), distance `5+` km (+0.0131), 3+ observers
  (+0.0130), 2 observers (+0.0101), duration `121+` minutes (+0.0095), and
  traveling checklists (+0.0076). This suggests the spatial residual is most
  useful where effort is high enough to reveal local assemblage/geography.
- The weakest stratum is short traveling/local movement, distance `(0,0.5]` km,
  where micro AUPRC drops by -0.0047. Stationary checklists and zero-distance
  checklists still gain slightly, but less than traveling/high-distance
  checklists.
- Macro AUPRC improves in every reported effort/spatial stratum, even where
  micro AUPRC is weak. This means species-level ranking is generally helped,
  but common species/checklist patterns can still drive local micro losses.
- Calibration remains the main tradeoff. ECE often worsens when ranking
  improves, especially for long duration (`121+`, delta ECE +0.0115), traveling
  checklists (+0.0052), and the large held-out block 44 (+0.0042). For a
  bias/effort model, these probability-calibration costs should not be ignored.
- Current interpretation: the spatial GNN appears to add real ranking signal in
  high-effort strata, but it may be using effort-correlated geography in a way
  that worsens probability calibration. The next validation should test whether
  this pattern holds under a changed spatial split.

Step 3 placeholder, alternate spatial split validation:

- Rebuild the graph dataset with at least one alternate spatial split seed or a
  different block granularity, then rerun only the primary residual GNN and the
  effort-strata diagnostics. The key question is whether the residual GNN still
  improves high-effort/traveling strata and whether the same species failures
  recur.
- Candidate variants:
  - same 8x8 blocks with a different `--split-seed`
  - finer blocks, such as 10x10, if test block selection remains balanced
  - coarser blocks only if the held-out geography remains representative
- Do not expand the graph architecture until the primary residual result is at
  least directionally stable under one changed held-out geography.
- Alternate split seed 37 graph dataset was built in
  `data/ebird/graph_top100_spatial_seed37` with the same top-100 species,
  features, 8x8 spatial blocks, test fraction, stratification species count,
  and negative sampling settings as the seed-19 graph dataset.
- The seed-37 graph dataset validated successfully and produced the same
  checklist/edge counts as seed 19: 661,979 checklists, 100 species, 529,526
  train checklists, 132,453 test checklists, 7,160,399 train positives, and
  1,857,102 test positives.
- The seed-37 primary residual spatial GNN is effectively identical to the
  seed-19 result: micro AUROC 0.8942, micro AUPRC 0.5927, macro AUROC 0.8482,
  macro AUPRC 0.4279, ECE 0.0084, and species calibration MAE 0.0145. The
  original seed-19 primary residual was micro AUROC 0.8944, micro AUPRC 0.5927,
  macro AUROC 0.8484, macro AUPRC 0.4287, ECE 0.0085, and species calibration
  MAE 0.0144.
- The seed-37 effort-strata diagnostics also replicate the same pattern:
  strongest micro-AUPRC gains in higher-effort contexts, including distance
  `(2,5]` km (+0.0136), distance `5+` km (+0.0132), 3+ observers (+0.0128),
  2 observers (+0.0096), duration `121+` minutes (+0.0096), and traveling
  checklists (+0.0075). The same weak stratum remains distance `(0,0.5]` km
  (-0.0048).
- Interpretation: changing only the split seed did not meaningfully change the
  validation problem. This is reassuring for reproducibility, but it is not a
  strong spatial-transfer stress test. The next split validation should change
  spatial block granularity, such as 10x10 blocks, rather than only changing the
  tie-break seed.

Completed seed-37 validation commands:

```
python exp/validate_ebird_graph_dataset.py --graph-dir data/ebird/graph_top100_spatial_seed37
```

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_seed37 --run-name spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001 --gnn-mode residual --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000
```

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_seed37 --spatial-run-name spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001
```

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial_seed37 --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial_seed37/spatial_gnn_baselines/spatial_gnn_spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001_test_species_metrics.csv --output data/ebird/graph_top100_spatial_seed37/spatial_gnn_baselines/residual_primary_graph_vs_tabular_species.csv
```

Recommended next stronger spatial validation, 10x10 blocks:

```
python exp/build_ebird_graph_dataset.py --processed-dir data/ebird/processed_nc_2020_2023 --output-dir data/ebird/graph_top100_spatial_10x10 --top-species 100 --feature-set both --split spatial-stratified --spatial-blocks-per-dim 10 --test-fraction 0.2 --stratify-species-count 20 --split-seed 19 --negative-ratio 5
```

```
python exp/validate_ebird_graph_dataset.py --graph-dir data/ebird/graph_top100_spatial_10x10
```

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001 --gnn-mode residual --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000
```

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001
```

10x10 spatial-block validation result:

- The 10x10 graph dataset built and validated successfully:
  661,979 checklists, 100 species, 524,959 train checklists, 137,020 test
  checklists, 7,149,369 train positives, and 1,868,132 test positives.
- The 10x10 primary residual spatial GNN is weaker on ranking than the 8x8
  split but much better calibrated: micro AUROC 0.8904, micro AUPRC 0.5763,
  macro AUROC 0.8428, macro AUPRC 0.4100, ECE 0.0014, max bin error 0.0072,
  and species calibration MAE 0.0129.
- Compared with the 8x8 seed-19 primary residual model, the 10x10 split loses
  about 0.0164 micro AUPRC and 0.0187 macro AUPRC, while ECE improves from
  0.0085 to 0.0014 and species calibration MAE improves from 0.0144 to 0.0129.
  This suggests the 10x10 held-out geography is a different and harder ranking
  problem, not just a rerun of the same validation setting.
- The effort-strata pattern is directionally similar but weaker. The spatial
  GNN still helps most in high-effort strata: duration `121+` minutes
  (+0.0118 micro AUPRC), distance `5+` km (+0.0113), distance `(2,5]` km
  (+0.0098), and traveling checklists (+0.0058). Stationary/zero-distance and
  short-duration strata are weaker, and block 79 is a clear local failure
  (-0.0127 micro AUPRC).
- Unlike the 8x8 split, the 10x10 effort-strata ECE deltas are mostly negative,
  meaning the spatial GNN improves or preserves calibration within most effort
  strata while losing some ranking. This changes the interpretation: the GNN is
  not simply overfitting effort geography, but its ranking advantage is
  sensitive to held-out block granularity.
- Current interpretation after 10x10: the spatial residual GNN is a credible
  component, but not yet a robust win. It improves calibration under the harder
  10x10 split and preserves the high-effort-strata signal, but the aggregate
  ranking gain seen in 8x8 does not transfer cleanly. The next diagnostic should
  compare 10x10 species-level deltas to the 8x8 species failures and identify
  whether the ranking drop is broad or concentrated in a few blocks/species.

10x10 species-level comparison command:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/spatial_gnn_spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001_test_species_metrics.csv --output data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/residual_primary_graph_vs_tabular_species.csv
```

10x10 species diagnostic command:

```
python exp/diagnose_ebird_spatial_gnn_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --comparison-csv data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/residual_primary_graph_vs_tabular_species.csv --boundary data/boundaries/nc_state_boundary.gpkg
```

Important comparability note for 10x10 species deltas:

- The first 10x10 species comparison above is exploratory only. It compares the
  10x10 graph species metrics against the existing tabular MLP species metrics
  in `data/ebird/baselines`, which were produced for the original 8x8
  spatial-stratified split. The `tabular_test_prevalence` and
  `graph_observed_rate` columns differ for many species, confirming that the
  compared test targets are not identical.
- The exploratory output is still useful for seeing which species are sensitive
  to the new held-out geography, but it should not be used as the final
  graph-vs-tabular species delta table.
- Before interpreting 10x10 species-level gains/losses, run a matching 10x10
  tabular MLP baseline into a separate output directory, then rerun the
  comparison against that directory.

Fair 10x10 tabular baseline command:

```
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --output-dir data/ebird/baselines_10x10 --top-species 100 --feature-set both --model mlp --split spatial-stratified --spatial-blocks-per-dim 10 --test-fraction 0.2 --stratify-species-count 20 --split-seed 19 --epochs 50
```

Fair 10x10 graph-vs-tabular species comparison command:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --baseline-dir data/ebird/baselines_10x10 --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/spatial_gnn_spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001_test_species_metrics.csv --output data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/residual_primary_graph_vs_tabular_species_fair_10x10.csv
```

Fair 10x10 species diagnostic command:

```
python exp/diagnose_ebird_spatial_gnn_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --comparison-csv data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/residual_primary_graph_vs_tabular_species_fair_10x10.csv --boundary data/boundaries/nc_state_boundary.gpkg
```

Fair 10x10 graph-vs-tabular result:

- The matching 10x10 MLP ecology + effort baseline ran successfully:
  macro AUROC 0.8404, macro AUPRC 0.4017, micro AUROC 0.8897, micro AUPRC
  0.5725, ECE 0.0051, max bin error 0.0201, and species calibration MAE
  0.0121.
- The 10x10 residual spatial GNN on the same all-pairs target has macro AUROC
  0.8428, macro AUPRC 0.4100, micro AUROC 0.8904, micro AUPRC 0.5763,
  ECE 0.0014, max bin error 0.0072, and species calibration MAE 0.0129.
- On the fair 10x10 comparison, the spatial GNN is a small but real aggregate
  improvement over tabular: +0.0007 micro AUROC, +0.0039 micro AUPRC,
  +0.0024 macro AUROC, and +0.0083 macro AUPRC. Calibration is better by ECE
  and max bin error, while species calibration MAE is slightly worse
  (+0.0008).
- This resolves the earlier comparability issue: `tabular_test_prevalence` and
  `graph_observed_rate` now match, so species-level AUPRC deltas are
  interpretable.

Exploratory 10x10 species diagnostic observations:

- The 10x10 diagnostic script ran successfully and wrote species diagnostics
  under
  `data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/diagnostics/species_diagnostics`.
  These outputs are useful for inspecting support and map geometry.
- The held-out species composition changed materially under 10x10. Examples:
  Eastern Towhee test prevalence rises to 0.3306, American Robin to 0.3143,
  Northern Mockingbird to 0.2605, while Red-headed Woodpecker drops to 0.0453.
  This reinforces that the 10x10 split is a different validation target.
- Some species with narrow coastal or water-associated geography have limited
  10x10 held-out support: Brown Pelican appears in only 2 held-out spatial
  blocks and 7 positive cells; Bufflehead and Pied-billed Grebe have only 17
  positive cells. These are likely sensitive to block geometry.
- The low-prevalence focus species still have enough checklists to inspect,
  but often only a small number of held-out blocks. Examples: Green Heron has
  3,860 positive held-out checklists across 3 blocks, 20 positive cells, and
  13 counties; Red-headed Woodpecker has 6,205 positives across 3 blocks,
  21 cells, and 15 counties; Brown Pelican has 7,008 positives but only
  2 blocks, 7 cells, and 3 counties. For these species, map interpretation
  should focus on whether failures are block-geometry artifacts, coastal
  transfer failures, or effort/protocol effects.
- The fair comparison shows the largest all-pairs AUPRC gains for
  Double-crested Cormorant (+0.0980), Killdeer (+0.0849),
  Black-and-white Warbler (+0.0830), White-eyed Vireo (+0.0828),
  Yellow-billed Cuckoo (+0.0675), Ring-billed Gull (+0.0619),
  Northern Parula (+0.0572), Indigo Bunting (+0.0533), Common Yellowthroat
  (+0.0522), and Yellow-throated Warbler (+0.0511).
- The fair comparison shows the largest all-pairs AUPRC losses for
  Red-headed Woodpecker (-0.1566), Eastern Meadowlark (-0.0711),
  House Sparrow (-0.0603), Bufflehead (-0.0362), Pied-billed Grebe (-0.0345),
  Red-winged Blackbird (-0.0249), Common Grackle (-0.0242), Swamp Sparrow
  (-0.0214), Laughing Gull (-0.0164), Tree Swallow (-0.0162), Wood Duck
  (-0.0150), and Blue-headed Vireo (-0.0143).
- The worst graph species calibration errors are Eastern Phoebe (0.0474),
  European Starling (0.0399), Northern Flicker (0.0396), House Finch (0.0383),
  Eastern Meadowlark (0.0375), Great Blue Heron (0.0346), Downy Woodpecker
  (0.0346), American Crow (0.0332), Northern Mockingbird (0.0325), Gray
  Catbird (0.0322), Eastern Towhee (0.0315), and American Robin (0.0310).
- Next diagnostic focus: inspect residual maps and block-level/effort-stratum
  behavior for the large fair losses, especially Red-headed Woodpecker,
  Eastern Meadowlark, House Sparrow, Bufflehead, Pied-billed Grebe, Swamp
  Sparrow, and Wood Duck. These losses matter because the aggregate GNN gain is
  modest and could hide species where spatial message passing is overcorrecting
  useful effort/ecology signal.

10x10 residual-map diagnostic command:

```
python exp/plot_ebird_spatial_gnn_residual_maps.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001 --species "Red-headed Woodpecker" "Eastern Meadowlark" "House Sparrow" "Bufflehead" "Pied-billed Grebe" "Swamp Sparrow" "Wood Duck" "Double-crested Cormorant" "Killdeer" "Black-and-white Warbler" --boundary data/boundaries/nc_state_boundary.gpkg
```

10x10 residual-map result:

- The residual-map diagnostic confirms that several large species losses are
  caused by broad negative spatial corrections, not merely small local ranking
  changes.
- The strongest down-corrections are for Red-headed Woodpecker, House Sparrow,
  Bufflehead, Eastern Meadowlark, Pied-billed Grebe, and Swamp Sparrow.
  Examples:
  - Red-headed Woodpecker mean probability drops from 0.1281 to 0.0587, with
    positive checklist mean delta -0.1218.
  - House Sparrow mean probability drops from 0.1000 to 0.0295, with positive
    checklist mean delta -0.1219.
  - Bufflehead mean probability drops from 0.0687 to 0.0318, with positive
    checklist mean delta -0.1769 and positive 90th-percentile absolute delta
    0.3061.
  - Eastern Meadowlark mean probability drops from 0.0673 to 0.0326, with
    positive checklist mean delta -0.0627.
- Wood Duck has a smaller mean correction than the other loss species
  (-0.0077 overall, -0.0194 on positives), so its AUPRC loss may be more about
  local ranking than broad suppression.
- Gain species do not all behave the same way. Double-crested Cormorant has a
  positive mean correction overall (+0.0142) but still a negative mean
  correction on positive checklists (-0.0223), implying that the gain may come
  from stronger suppression of false-positive areas rather than uniformly
  increasing known positives. Black-and-white Warbler has only a small mean
  correction (-0.0051), consistent with a more targeted residual.
- Interpretation: the residual spatial GNN is useful, but the current residual
  head can over-suppress held-out geography for some species. This points
  toward adding constraints or diagnostics before adding richer graph
  relations: species-specific residual shrinkage, gated residual strength,
  calibration-aware validation, and block/species residual audits.

Next 10x10 gated-residual check:

- Run the same 10x10 model with a gated residual head. The goal is not just to
  improve aggregate metrics; it is to test whether the model can keep useful
  spatial corrections while reducing the broad negative corrections seen for
  Red-headed Woodpecker, House Sparrow, Bufflehead, Eastern Meadowlark,
  Pied-billed Grebe, and Swamp Sparrow.

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_gated_h128_l2_z128_cell64_cl1_wd0p0001_gbm2 --gnn-mode gated --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --gate-init-bias -2
```

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --baseline-dir data/ebird/baselines_10x10 --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/spatial_gnn_spatial_gcn_gated_h128_l2_z128_cell64_cl1_wd0p0001_gbm2_test_species_metrics.csv --output data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/gated_gbm2_graph_vs_tabular_species_fair_10x10.csv
```

```
python exp/plot_ebird_spatial_gnn_residual_maps.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_gated_h128_l2_z128_cell64_cl1_wd0p0001_gbm2 --species "Red-headed Woodpecker" "Eastern Meadowlark" "House Sparrow" "Bufflehead" "Pied-billed Grebe" "Swamp Sparrow" "Wood Duck" "Double-crested Cormorant" "Killdeer" "Black-and-white Warbler" --boundary data/boundaries/nc_state_boundary.gpkg
```

10x10 gated-residual result:

- The gated residual model did not improve the 10x10 primary residual model.
  Aggregate metrics were slightly worse than the residual model: micro AUROC
  0.8892 vs 0.8904, micro AUPRC 0.5755 vs 0.5763, macro AUROC 0.8421 vs
  0.8428, macro AUPRC 0.4095 vs 0.4100. Calibration also worsened: ECE 0.0080
  vs 0.0014 and species calibration MAE 0.0169 vs 0.0129.
- The gate helped one key failure species: Red-headed Woodpecker changed from a
  broad negative residual to a near-neutral/slightly positive residual
  correction. Its all-pairs AUPRC loss improved from -0.1566 under the
  residual model to -0.0895 under the gated model, but it remains the largest
  species loss.
- The gate made several other residual suppressions worse. Examples:
  - Eastern Meadowlark mean probability delta -0.1522 and positive mean delta
    -0.2102; AUPRC loss -0.0298.
  - House Sparrow mean probability delta -0.1296 and positive mean delta
    -0.1528; AUPRC loss -0.0760.
  - Bufflehead mean probability delta -0.0810 and positive mean delta -0.2757;
    AUPRC loss -0.0359.
  - Pied-billed Grebe mean probability delta -0.1302 and positive mean delta
    -0.3347; AUPRC loss -0.0608.
  - Killdeer remains an AUPRC gain (+0.0905), but the residual map shows a
    large negative correction on positives (-0.1725), so its gain likely comes
    from stronger suppression of false-positive areas rather than better
    absolute detection probabilities.
- Interpretation: a simple global gate is not enough. It can change which
  species are over-suppressed, but it does not reliably constrain the residual
  to helpful spatial corrections. The next modeling change should target
  species-specific residual shrinkage or residual calibration directly, rather
  than adding more spatial capacity.

Species-specific residual scale:

- `exp/ebird_spatial_gnn_baseline.py` now supports a species-specific residual
  scale for residual and gated spatial GNN modes.
- The model form is:

  \[
  \operatorname{logit} P(y_{c,j}=1)
  =
  \text{base}_{j}(x_c)
  +
  \alpha_j r_j(g_c)
  \]

  where \(r_j(g_c)\) is the spatial-cell residual for species \(j\) at the
  checklist's grid cell, and \(\alpha_j\) is a learned species-specific scale.
- The first recommended version uses `--species-residual-scale sigmoid`, which
  constrains each \(\alpha_j\) to `(0, 1)`, initializes it at 0.25, and adds an
  explicit L2 penalty on the effective scale. This lets species that benefit
  from spatial correction use it, while pushing harmful residuals toward the
  base ecology + effort model.
- The scale parameter is excluded from AdamW weight decay and controlled by
  `--species-residual-scale-l2`, because weight decay on the raw sigmoid logit
  is not equivalent to shrinking the effective residual scale.

10x10 species-scaled residual command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_residual_scaled_sigmoid025_l2_0p001 --gnn-mode residual --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.25 --species-residual-scale-l2 0.001
```

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --baseline-dir data/ebird/baselines_10x10 --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/spatial_gnn_spatial_gcn_residual_scaled_sigmoid025_l2_0p001_test_species_metrics.csv --output data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/residual_scaled_sigmoid025_l2_0p001_graph_vs_tabular_species_fair_10x10.csv
```

```
python exp/plot_ebird_spatial_gnn_residual_maps.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_residual_scaled_sigmoid025_l2_0p001 --species "Red-headed Woodpecker" "Eastern Meadowlark" "House Sparrow" "Bufflehead" "Pied-billed Grebe" "Swamp Sparrow" "Wood Duck" "Double-crested Cormorant" "Killdeer" "Black-and-white Warbler" --boundary data/boundaries/nc_state_boundary.gpkg
```

10x10 species-scaled residual result:

- This is the best 10x10 aggregate model so far: micro AUROC 0.8905, micro
  AUPRC 0.5785, macro AUROC 0.8426, macro AUPRC 0.4110, ECE 0.0020, max bin
  error 0.0073, and species calibration MAE 0.0127.
- Relative to the unscaled residual model, the scaled residual improves micro
  AUPRC from 0.5763 to 0.5785, macro AUPRC from 0.4100 to 0.4110, and species
  calibration MAE from 0.0129 to 0.0127, while ECE worsens slightly from
  0.0014 to 0.0020. It remains much better calibrated than the gated model.
- Learned species residual scales did not behave as a simple automatic
  fail-safe. The mean scale increased from the 0.25 initialization to 0.3289,
  with p10 0.2816, median 0.3229, p90 0.3773, and max 0.4848. Several failure
  species learned large scales: Red-headed Woodpecker 0.4502, Eastern
  Meadowlark 0.4368, Pied-billed Grebe 0.3980. This means the model still
  trusts spatial correction for some species where held-out ranking suffers.
- The residual maps show mixed improvements:
  - Black-and-white Warbler is close to ideal for this mechanism: mean delta
    -0.0000, positive mean delta -0.0040, and AUPRC gain +0.0941.
  - Swamp Sparrow and Wood Duck are now close to neutral residual corrections,
    and their losses shrink relative to the unscaled residual.
  - Eastern Meadowlark suppression is less severe than the gated model but
    still harmful: positive mean delta -0.0350 and AUPRC loss -0.0491.
  - Red-headed Woodpecker and House Sparrow remain major failures with broad
    negative corrections: Red-headed Woodpecker positive mean delta -0.1282 and
    AUPRC loss -0.1612; House Sparrow positive mean delta -0.1216 and AUPRC
    loss -0.0810.
  - Bufflehead remains strongly suppressed on positives (-0.1898) and loses
    AUPRC (-0.0444).
- Interpretation: species-specific scaling is useful as regularization and is
  the best aggregate 10x10 model so far, but it is not sufficient as a safety
  mechanism. The next constraint should penalize the residual values
  themselves, especially on positive training examples or by species, instead
  of only penalizing the scale parameter.

10x10 stronger species-scale shrinkage result:

- A stronger shrinkage run with `--species-residual-scale-init 0.10` and
  `--species-residual-scale-l2 0.01` is now the best aggregate 10x10 model so
  far: micro AUROC 0.8917, micro AUPRC 0.5808, macro AUROC 0.8439, macro AUPRC
  0.4126, ECE 0.0033, max bin error 0.0127, and species calibration MAE
  0.0118.
- Compared with the previous scaled run, it improves micro AUPRC by +0.0023,
  macro AUPRC by +0.0015, and species calibration MAE by -0.0009. ECE worsens
  from 0.0020 to 0.0033 but remains better than the original tabular MLP
  baseline ECE of 0.0051.
- The learned scales are now much more constrained: mean 0.1338, min 0.0794,
  p10 0.1067, median 0.1309, p90 0.1639, and max 0.2098. This is a better
  residual-control regime than the 0.25-initialized run, where the mean scale
  grew to 0.3289 and several failure species retained large scales.
- This result suggests the useful spatial signal is relatively small and that
  aggressive residual capacity was causing some of the earlier species-level
  failures. The next check is whether the better aggregate result also improves
  the known failure species in the fair species comparison and residual maps.

10x10 stronger species-scale comparison command:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --baseline-dir data/ebird/baselines_10x10 --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/spatial_gnn_spatial_gcn_residual_scaled_sigmoid010_l2_0p01_test_species_metrics.csv --output data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/residual_scaled_sigmoid010_l2_0p01_graph_vs_tabular_species_fair_10x10.csv
```

10x10 stronger species-scale residual-map command:

```
python exp/plot_ebird_spatial_gnn_residual_maps.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_residual_scaled_sigmoid010_l2_0p01 --species "Red-headed Woodpecker" "Eastern Meadowlark" "House Sparrow" "Bufflehead" "Pied-billed Grebe" "Swamp Sparrow" "Wood Duck" "Double-crested Cormorant" "Killdeer" "Black-and-white Warbler" --boundary data/boundaries/nc_state_boundary.gpkg
```

10x10 stronger species-scale species diagnostics:

- The fair species comparison confirms that tighter residual scaling improves
  the aggregate model while reducing, but not eliminating, over-suppression.
- Largest AUPRC gains over tabular are concentrated in species where modest
  spatial structure is useful: White-eyed Vireo (+0.0930),
  Black-and-white Warbler (+0.0929), Ring-billed Gull (+0.0914),
  Double-crested Cormorant (+0.0776), Killdeer (+0.0723),
  Yellow-billed Cuckoo (+0.0652), Yellow-throated Warbler (+0.0613),
  Great Black-backed Gull (+0.0545), Indigo Bunting (+0.0517), Common
  Yellowthroat (+0.0492), Northern Parula (+0.0489), and American Robin
  (+0.0483).
- Largest AUPRC losses remain species-specific and should not be ignored:
  Red-headed Woodpecker (-0.1632), House Sparrow (-0.1108), Bufflehead
  (-0.0275), Eastern Meadowlark (-0.0260), Common Grackle (-0.0252),
  European Starling (-0.0201), Brown Pelican (-0.0191), Northern Mockingbird
  (-0.0190), Red-winged Blackbird (-0.0188), Northern Rough-winged Swallow
  (-0.0184), Green Heron (-0.0129), and Mourning Dove (-0.0126).
- Residual maps show the tighter scale made several corrections more moderate:
  - Eastern Meadowlark is nearly neutral on positives now (positive mean delta
    -0.0021), but still loses AUPRC (-0.0260), suggesting a ranking/geography
    issue rather than broad positive suppression.
  - Swamp Sparrow and Wood Duck now get positive corrections on positives
    (+0.0519 and +0.0224) and their losses are smaller.
  - Black-and-white Warbler is almost neutral in mean probability delta
    (+0.0005 on positives) while gaining strongly in AUPRC (+0.0929), which is
    close to the desired behavior: graph signal improves ranking without
    broadly changing probabilities.
- Persistent failures:
  - Red-headed Woodpecker is still strongly down-corrected on positives
    (-0.0886) and remains the largest AUPRC loss (-0.1632).
  - House Sparrow is still broadly down-corrected (-0.0888 overall, -0.1355 on
    positives) and loses AUPRC (-0.1108).
  - Bufflehead remains down-corrected on positives (-0.1411), though less than
    under weaker shrinkage.
- Interpretation: the current best model supports the main project goal: a
  graph component can add useful spatial context beyond ecology + effort while
  retaining good calibration. The remaining issue is not "does the graph help?"
  but "how do we prevent graph correction from harming a subset of species?"
  The next diagnostic should inspect whether the persistent failures are tied
  to specific held-out blocks, effort strata, or baseline-vs-residual ranking
  inversions.

10x10 stronger species-scale effort-strata diagnostics:

- The stronger scaled residual model improves almost every effort stratum
  against the retrained tabular MLP.
- Largest micro-AUPRC gains are in higher-effort settings: duration `121+`
  minutes (+0.0145), distance `5+` km (+0.0135), distance `(2,5]` km
  (+0.0128), two observers (+0.0103), traveling checklists (+0.0102), and
  duration `61-120` minutes (+0.0088).
- Unlike earlier variants, short/low-effort strata are no longer obvious
  failures: stationary checklists gain +0.0028 micro AUPRC, zero-distance
  checklists gain +0.0029, duration `1-10` minutes gains +0.0043, and distance
  `(0,0.5]` km gains +0.0073. Their ECE deltas are also generally negative,
  meaning calibration improves.
- The remaining weakness is spatial-block specific. Block 65 is a clear win
  (+0.0115 micro AUPRC, +0.0122 macro AUPRC), block 31 has a small micro gain
  but macro loss (+0.0044 micro AUPRC, -0.0031 macro AUPRC), and block 79 is a
  clear local failure (-0.0070 micro AUPRC, -0.0116 macro AUPRC).
- Interpretation: the scaled residual model is no longer mainly failing by
  protocol/duration/distance effort strata. The next validation target should
  be block-by-species behavior, especially block 79 and the persistent species
  losses: Red-headed Woodpecker, House Sparrow, Bufflehead, Eastern
  Meadowlark, Common Grackle, and European Starling.

Block-by-species diagnostic:

- `exp/diagnose_ebird_block_species.py` compares the current spatial GNN with a
  retrained tabular MLP within each held-out spatial block and species. This is
  intended as a framework diagnostic rather than tuning to the NC/top-100
  species list.
- The goal is to classify failures into general model behaviors:
  - spatial residual over-suppresses valid positives in a held-out geography
  - graph signal improves false-positive suppression without broad probability
    shifts
  - species failures are concentrated in one held-out block
  - species failures persist across blocks, suggesting species/process-specific
    issues rather than local geometry
  - losses occur in low-support block/species combinations and may reflect
    validation granularity
- Outputs:
  - `block_species_metrics.csv`: one row per held-out block and species,
    including tabular vs spatial AUROC/AUPRC/calibration deltas and residual
    probability summaries.
  - `block_summary.csv`: block-level counts and mean species deltas.
  - `block_species_summary.png`: compact visual summary of block-level behavior.
  - `block_species_metadata.json`: run configuration and spatial summary.

Recommended command:

```
python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_residual_scaled_sigmoid010_l2_0p01
```

Block-by-species diagnostic result:

- The diagnostic confirms that the remaining weakness is not primarily a broad
  effort-stratum problem. It is spatial-transfer behavior that varies by held
  out block and species.
- Block 65 is where the graph framework is working best: mean species AUPRC
  delta +0.0124, median +0.0065, 68 species with AUPRC gain and 27 with loss.
  Large gains include Black-and-white Warbler (+0.1533), White-eyed Vireo
  (+0.1031), Bald Eagle (+0.0901), Yellow-throated Warbler (+0.0727),
  Yellow-billed Cuckoo (+0.0718), Gray Catbird (+0.0692), Eastern Towhee
  (+0.0637), and American Robin (+0.0629).
- Block 79 is the clearest local transfer failure: mean species AUPRC delta
  -0.0120, median -0.0044, 56 species with AUPRC loss and 41 with gain. The
  largest failures are severe: House Sparrow (-0.3371), House Finch (-0.2666),
  European Starling (-0.1457), Red-headed Woodpecker (-0.1365), Northern
  Mockingbird (-0.1093), Mourning Dove (-0.1031), Northern Cardinal (-0.0802),
  American Robin (-0.0689), and Red-winged Blackbird (-0.0615).
- Block 31 is mixed: mean species AUPRC delta -0.0031, median +0.0012, 45
  species with loss and 50 with gain. The most important losses include Eastern
  Meadowlark (-0.1644), Ovenbird (-0.0846), and Mallard (-0.0711).
- Several losses are true residual down-correction failures: House Sparrow in
  block 79 has positive mean delta -0.2253, House Finch in block 79 -0.2716,
  Red-headed Woodpecker in block 79 -0.2314, Red-headed Woodpecker in block 65
  -0.0802, Eastern Meadowlark in block 31 -0.0959, and Bufflehead in block 65
  -0.0588.
- Some losses are not simple over-suppression. Mourning Dove in block 79 loses
  AUPRC despite positive mean delta on positives (+0.0335), and Red-winged
  Blackbird in block 79 loses AUPRC despite a large positive mean delta on
  positives (+0.2152). These are ranking/overprediction failures, not just
  probability suppression.
- Framework interpretation: spatial message passing can help substantially, but
  it is sensitive to held-out geography. The current residual-scale constraint
  controls aggregate behavior, yet block 79 shows that a spatial graph residual
  can still learn the wrong correction for transferred urban/generalist
  assemblages and some woodpecker/sparrow species. The next framework step
  should be a diagnostic or model component that distinguishes ecological
  spatial structure from observation-geography residuals, rather than tuning
  around individual species.

Separated suitability/bias architecture:

- `exp/ebird_spatial_gnn_baseline.py` now supports
  `--component-mode separated` as a first-pass implementation of the core
  project goal: model species suitability separately from effort/observer
  geography bias.
- This is still a detection model on complete checklists, but its logit is
  decomposed into:

  \[
  \operatorname{logit} P(y_{c,j}=1)
  =
  \text{suitability}_{j}(x_{\text{ecology},c})
  +
  \text{bias}_{j}(x_{\text{effort},c}, g_c)
  +
  \alpha_j r_j(g_c)
  \]

  where \(g_c\) is the checklist's spatial-cell context from the GCN and
  \(\alpha_j r_j(g_c)\) is the existing species-scaled residual correction.
- Current feature split:
  - Suitability/ecology path: day-of-year sine/cosine, canopy, elevation,
    distance to waterbody, and distance to coastline.
  - Bias/effort path: x/y location, day-of-week sine/cosine, duration, effort
    distance, number of observers, traveling indicator, and spatial-cell GCN
    context.
- The point of this split is not to claim that x/y is never ecological or that
  every covariate is perfectly assigned. It is a testable first decomposition:
  environmental/seasonal variables carry species suitability, while protocol,
  effort, accessibility/observer geography, and checklist-density spatial
  context carry sampling/detectability bias.
- The same species residual scale is retained, but now the residual is
  evaluated against a model that already has an explicit bias path. If the
  residual still causes block-specific failures, that is stronger evidence that
  the residual is learning non-transferable geography rather than useful
  suitability.

10x10 separated suitability/bias command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_separated_scaled_sigmoid010_l2_0p01 --gnn-mode residual --component-mode separated --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01
```

Initial separated suitability/bias result:

- The first separated architecture underperforms both the tabular MLP and the
  joint scaled spatial GNN: micro AUROC 0.8845, micro AUPRC 0.5631, macro AUROC
  0.8354, macro AUPRC 0.3984, ECE 0.0061, max bin error 0.0246, and species
  calibration MAE 0.0157.
- Compared with the joint scaled model, this is a large drop: micro AUPRC
  0.5808 to 0.5631 and macro AUPRC 0.4126 to 0.3984.
- Interpretation: separating suitability and bias is still the correct
  framework goal, but this naive split is too rigid. It removes useful
  interactions between environmental, spatial, seasonal, and effort variables
  that the joint model was using. The current feature assignment is also only a
  first approximation; x/y can carry ecological range structure as well as
  observer geography, and season can affect both availability and detectability.
- Do not treat this as evidence against the decomposition. Treat it as evidence
  that the decomposition needs a shared trunk, interaction terms, or weakly
  constrained components rather than a hard split into two independent MLPs.

Shared-trunk suitability/bias architecture:

- `exp/ebird_spatial_gnn_baseline.py` now supports
  `--component-mode shared`.
- This keeps the original full-covariate checklist encoder as a shared trunk,
  then adds a separate effort/bias head conditioned on:
  - the shared checklist latent representation
  - effort/access features: x/y, day of week, duration, distance, observers,
    and traveling indicator
  - spatial-cell GCN context
- The suitability path remains the original species-embedding/direct species
  head from the shared latent representation. The effort/bias path adds a
  second checklist/species contribution. The scaled spatial residual remains
  available as a constrained correction.
- This is the better framework test than the hard split because it still allows
  interactions among ecology, season, space, and effort while making the
  bias/detectability contribution explicit.

10x10 shared-trunk suitability/bias command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_shared_scaled_sigmoid010_l2_0p01 --gnn-mode residual --component-mode shared --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01
```

Initial shared-trunk suitability/bias result:

- The shared-trunk decomposition also underperforms the joint scaled model:
  micro AUROC 0.8849, micro AUPRC 0.5606, macro AUROC 0.8376, macro AUPRC
  0.3988, ECE 0.0103, max bin error 0.0506, and species calibration MAE
  0.0195.
- It is slightly worse than the hard separated model on micro AUPRC and
  calibration, though slightly better on macro AUROC/AUPRC. Both are clearly
  worse than the joint scaled residual model.
- Interpretation: simply adding an explicit effort/bias head is not enough.
  In this form, the bias head adds unconstrained capacity and worsens
  calibration/transfer instead of cleanly separating detectability from
  suitability.
- Current framework conclusion: keep the joint scaled residual as the working
  baseline. The suitability/bias decomposition remains the right goal, but it
  likely needs identification constraints rather than just architectural
  separation. Candidate constraints include:
  - bias head predicts checklist-level detection propensity shared across
    species, with only low-rank species deviations
  - ecological/suitability head is evaluated under standardized effort
    scenarios
  - effort/bias head is regularized by protocol/duration/distance calibration
    targets
  - spatial graph context is split into environmental-neighbor context and
    observer/accessibility context
  - observer/locality/checklist-density terms are isolated from species
    suitability terms and stress-tested under blocked validation

Constrained effort/access bias component:

- `exp/ebird_spatial_gnn_baseline.py` now supports `--effort-bias-mode` as a
  more constrained way to model observation geography separately from the
  species suitability path.
- This keeps `--component-mode joint` as the working baseline and adds a
  zero-initialized effort/access bias component using:
  - x/y location
  - day of week
  - duration
  - effort distance
  - number of observers
  - traveling indicator
  - spatial-cell GCN context
- Two versions are available:
  - `--effort-bias-mode shared`: one checklist-level logit adjustment shared by
    all species. This is the strongest identification constraint and tests
    whether effort/access geography mostly acts as checklist-level detection
    propensity.
  - `--effort-bias-mode lowrank`: a shared checklist propensity plus low-rank
    species deviations. This allows species-specific detectability differences
    without giving every species a fully independent free bias head.
- Both versions are initialized as no-ops, so they start from the joint scaled
  residual model and only learn bias structure if it improves the training
  objective. In the low-rank version, the checklist-side low-rank head is
  zero-initialized while the species low-rank embeddings are initialized with
  small nonzero values; this keeps the initial low-rank output at zero while
  still allowing gradients to move the species-specific bias directions.
  `--effort-bias-l2` penalizes the effort/access bias logits directly in each
  training batch.
- This is a better test of the overall framework goal than the hard split or
  unconstrained shared head: it separates effort/access bias while limiting how
  much that component can absorb species suitability.

10x10 shared effort-bias command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_joint_scaled_effort_shared_l2_0p001 --gnn-mode residual --component-mode joint --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --effort-bias-mode shared --effort-bias-l2 0.001
```

10x10 low-rank effort-bias command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_joint_scaled_effort_lowrank8_l2_0p001 --gnn-mode residual --component-mode joint --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --effort-bias-mode lowrank --effort-bias-rank 8 --effort-bias-l2 0.001
```

Initial 10x10 constrained effort-bias results:

- Shared effort-bias:
  - micro AUROC 0.8903
  - micro AUPRC 0.5783
  - macro AUROC 0.8439
  - macro AUPRC 0.4109
  - ECE 0.0044
  - max bin error 0.0228
  - species calibration MAE 0.0157
- Low-rank effort-bias:
  - micro AUROC 0.8890
  - micro AUPRC 0.5709
  - macro AUROC 0.8416
  - macro AUPRC 0.4054
  - ECE 0.0031
  - max bin error 0.0128
  - species calibration MAE 0.0142
- Interpretation:
  - Neither constrained effort-bias run beats the current joint scaled residual
    model on AUPRC. The current reference remains
    `spatial_gcn_residual_scaled_sigmoid010_l2_0p01`, with micro AUPRC 0.5808,
    macro AUPRC 0.4126, ECE 0.0033, and species calibration MAE 0.0118.
  - The shared effort-bias run is close to the reference in ranking but has
    weaker calibration. This suggests the added shared bias component is not
    clearly buying enough separation yet.
  - The low-rank effort-bias run has the lowest ECE and max-bin calibration
    error of these three, but its AUPRC drop is larger. That looks like a
    regularization/calibration tradeoff rather than an immediate model upgrade.
  - The constrained-bias idea remains aligned with the framework goal, but this
    first implementation should be treated as a diagnostic branch. The next
    check is whether either run improves effort-strata behavior or block-level
    failure modes enough to justify further refinement.

Fair species comparison and shared effort-bias diagnostics:

- Shared effort-bias vs 10x10 tabular MLP:
  - Clear species-level AUPRC gains include White-eyed Vireo (+0.0754),
    Killdeer (+0.0669), Ring-billed Gull (+0.0639), Double-crested Cormorant
    (+0.0595), Yellow-throated Warbler (+0.0570), Great Black-backed Gull
    (+0.0519), American Robin (+0.0502), Indigo Bunting (+0.0487), Common
    Yellowthroat (+0.0480), and Field Sparrow (+0.0478).
  - The largest losses remain the same basic failure species: Red-headed
    Woodpecker (-0.1206 AUPRC), House Sparrow (-0.1001), Bufflehead (-0.0345),
    Red-winged Blackbird (-0.0320), and several water/edge or urban-associated
    species.
  - Species calibration errors are still nontrivial for common/widespread
    species including American Crow (0.0638), Blue Jay (0.0492), European
    Starling (0.0465), Eastern Phoebe (0.0445), and House Finch (0.0421).
- Low-rank effort-bias vs 10x10 tabular MLP:
  - It improves some species that likely benefit from species-specific
    detectability/bias flexibility, including Black-and-white Warbler (+0.0874
    AUPRC), Ring-billed Gull (+0.0850), American Robin (+0.0632), American
    Herring Gull (+0.0587), Double-crested Cormorant (+0.0585), White-eyed
    Vireo (+0.0568), and Great Black-backed Gull (+0.0550).
  - The same hard failures remain, and some worsen: Red-headed Woodpecker
    (-0.1681), House Sparrow (-0.0827), Tree Swallow (-0.0687), Canada Goose
    (-0.0450), Bufflehead (-0.0426), Great Egret (-0.0356), and Pied-billed
    Grebe (-0.0339).
  - Interpretation: low-rank bias is not simply better than shared bias. It can
    improve some guilds/species but appears to amplify the same local/species
    failure modes that already motivated this diagnostic branch.
- Shared effort-bias effort-strata diagnostics:
  - It continues to improve longer and higher-effort traveling checklists:
    distance 5+ (+0.0139 micro AUPRC), distance (2,5] (+0.0131), duration 121+
    (+0.0114), observers 3+ (+0.0092), and traveling protocol (+0.0088).
  - It is weak or negative in the low-effort/stationary strata: stationary
    protocol (-0.0021), zero-distance (-0.0021), and short duration only
    slightly positive (+0.0015).
  - Block 79 remains the main spatial failure and is worse than in the current
    reference model: -0.0157 micro AUPRC and -0.0095 macro AUPRC vs tabular.
    Block 65 remains positive (+0.0082 micro), and block 31 is mildly positive
    (+0.0072 micro) but less convincing by macro AUPRC.
- Shared effort-bias block/species diagnostics:
  - Block 65 is still where the graph correction works best on average:
    mean delta AUPRC +0.0097, median +0.0066, 66 species with gains and 29
    with losses.
  - Block 79 remains a broad local transfer problem: mean delta AUPRC -0.0098,
    48 losses and 49 gains, with large losses for House Sparrow (-0.3134),
    House Finch (-0.2682), Northern Mockingbird (-0.1276), European Starling
    (-0.1173), Mourning Dove (-0.1056), and Red-winged Blackbird (-0.0968).
  - Block 31 is mixed but no longer broadly bad under shared effort-bias:
    mean delta AUPRC +0.0022, median +0.0039, 62 gains and 33 losses. Important
    losses still include Eastern Meadowlark (-0.0907), Mallard (-0.0718), and
    Ovenbird (-0.0593).
- Decision:
  - The shared effort-bias component does not solve the core block 79 failure
    and slightly weakens aggregate performance. It should not replace the
    current joint scaled residual reference.
  - The constrained effort-bias branch is still useful because it clarifies that
    naive effort/access separation is not enough. The next framework change
    should focus on *where* the bias/suitability distinction is identified:
    spatial block transfer, local urban/coastal effort structure, and species
    whose positives are concentrated in the problematic held-out blocks.
- Low-rank effort-bias effort-strata diagnostics:
  - The low-rank model is much less useful by effort stratum than the shared
    effort-bias model or the current joint scaled residual reference.
  - It has only small gains in high-effort strata: distance 5+ (+0.0066 micro
    AUPRC), duration 121+ (+0.0042), observers 3+ (+0.0040), distance (2,5]
    (+0.0032), and traveling protocol (+0.0009).
  - It loses performance in zero-distance/stationary strata: zero-distance
    (-0.0055), stationary protocol (-0.0055), and duration 31-60 (-0.0056).
  - Spatial block deltas are effectively flattened rather than improved:
    block 79 is only +0.0004 micro AUPRC but -0.0036 macro AUPRC; block 65 is
    -0.0002 micro but +0.0057 macro; block 31 is -0.0027 micro and -0.0063
    macro.
- Low-rank effort-bias block/species diagnostics:
  - Block 65 remains the only clearly positive block on average: mean delta
    AUPRC +0.0058, median +0.0046, 58 species gains and 37 losses.
  - Block 79 is less negative on average than the shared effort-bias model
    (mean -0.0040 vs -0.0098), but the hard urban/generalist failures remain:
    House Finch (-0.2673), House Sparrow (-0.2555), Mourning Dove (-0.0928),
    European Starling (-0.0899), and Northern Mockingbird (-0.0884).
  - Block 31 becomes worse than under shared effort-bias: mean delta AUPRC
    -0.0064, with important losses for Mallard (-0.1183), Eastern Meadowlark
    (-0.1066), Tree Swallow (-0.1039), Pied-billed Grebe (-0.0920), Canada
    Goose (-0.0831), Belted Kingfisher (-0.0785), and Ovenbird (-0.0698).
  - Interpretation: the low-rank bias component mostly smooths/regularizes the
    predictions. It improves some calibration summaries, but it does not create
    a better bias/suitability separation and does not solve the local transfer
    failures.
- Updated decision:
  - Stop treating `--effort-bias-mode shared` or `lowrank` as candidate lead
    architectures for now.
  - Keep the current lead model as the joint scaled residual spatial GNN:
    `spatial_gcn_residual_scaled_sigmoid010_l2_0p01`.
  - Use the constrained-bias results as evidence that the next framework step
    should not simply add another effort head. It should make the spatial
    correction more identifiable by validating or constraining *local transfer*:
    when a spatial correction helps in observed effort-heavy areas but fails in
    held-out blocks, the model needs a way to distinguish ecological spatial
    signal from observer/access geography.

Cell-level residual-vs-effort diagnostic:

- Added `exp/diagnose_ebird_cell_residual_effort.py` to test whether the
  spatial GNN residual correction is behaving more like ecological suitability
  or observer/access bias.
- The script evaluates a saved residual/gated spatial GNN on held-out
  checklists, aggregates by spatial GNN cell, and writes:
  - `*_cell_summary.csv`: cell-level correction, effort/access, ecology, and
    observed-prevalence summaries.
  - `*_cell_correlations.csv`: Pearson and Spearman correlations between
    residual-correction summaries and predictors such as checklist density,
    observed rate, protocol mix, duration, distance, observer/locality
    concentration, canopy, elevation, and water/coast distance.
  - `cell_residual_effort_scatter.png`: quick scatter diagnostics for residual
    magnitude/direction vs effort/access variables.
  - `cell_residual_effort_maps.png`: maps of mean residual correction, mean
    absolute residual correction, and held-out checklist density.
- Interpretation:
  - If `probability_delta_abs_mean` or `probability_delta_mean` is strongly
    associated with `log_checklists`, `traveling_rate`, `stationary_rate`,
    `observer_per_checklist`, or `locality_per_checklist`, then the residual
    correction is likely absorbing observation/access geography.
  - If residual corrections are more associated with observed prevalence,
    canopy/elevation/water/coast gradients, and are not concentrated only in
    high-effort cells, that is more consistent with ecological spatial signal.
  - If high residual magnitudes align with the known failed blocks/species, the
    next architecture should constrain spatial residuals by transfer reliability
    rather than by adding another free effort head.

Current lead-model command:

```
python exp/diagnose_ebird_cell_residual_effort.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_residual_scaled_sigmoid010_l2_0p01
```

Initial lead-model cell residual/effort diagnostic result:

- The diagnostic summarized 26 held-out spatial cells using the default
  `--min-cell-checklists 25` threshold.
- The largest mean absolute residual corrections are concentrated in held-out
  cells from blocks 79 and 31, which are also the blocks that have driven many
  of the block/species failures.
- The strongest associations show that the spatial residual correction is not
  purely ecological:
  - `probability_delta_abs_mean` is strongly positively associated with
    `traveling_rate` (Spearman 0.6609), `effort_distance_km_mean` (0.5822),
    `effort_distance_km_p90` (0.5325), `number_observers_mean` (0.4947), and
    `observer_per_checklist` (0.4701).
  - The same residual magnitude is strongly negatively associated with
    `stationary_rate` (-0.6609).
  - This means the spatial GNN correction is largest in cells dominated by
    traveling, higher-effort, higher-observer activity. That is a warning sign
    for observer/access bias absorption.
- The residual is not only effort/access bias:
  - `probability_delta_mean` is associated with `canopy_median_mean` (0.6554),
    `distance_to_waterbody_m_mean` (0.5419), observed rate (0.4667), and
    distance to coastline (0.4503).
  - Positive-only residual corrections are associated with distance to
    coastline (0.6609), distance to waterbody (0.6062), and elevation (0.5009).
  - Negative-only residual corrections are associated with canopy (0.6643) and
    distance to waterbody (0.5084).
- Interpretation:
  - The current lead residual GNN is learning a mixture of ecological spatial
    signal and effort/access geography.
  - The spatial correction is probably useful because it picks up real
    environmental/geographic structure, but it is not yet an identifiable
    suitability-only correction.
  - This explains why aggregate metrics improve while specific held-out blocks
    and species fail: the residual can transfer poorly where observer/access
    geography differs from training cells.
- Next diagnostic refinement:
  - Re-run this diagnostic with a lower cell threshold to test whether the
    correlation pattern is robust beyond the 26 best-supported cells.
  - Compare the lead residual model against the constrained effort-bias runs in
    the same cell-level diagnostic. If the constrained-bias models reduce the
    residual/effort correlations but also reduce AUPRC, then we have confirmed a
    bias-suitability tradeoff rather than just random noise.

Robustness command with smaller cells:

```
python exp/diagnose_ebird_cell_residual_effort.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_residual_scaled_sigmoid010_l2_0p01 --min-cell-checklists 10
```

Shared effort-bias cell diagnostic:

```
python exp/diagnose_ebird_cell_residual_effort.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_joint_scaled_effort_shared_l2_0p001 --min-cell-checklists 10
```

Low-rank effort-bias cell diagnostic:

```
python exp/diagnose_ebird_cell_residual_effort.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_joint_scaled_effort_lowrank8_l2_0p001 --min-cell-checklists 10
```

Cell diagnostic robustness at `--min-cell-checklists 10`:

- The robustness run summarized 28 held-out cells, only two more than the
  default threshold, so the previous 26-cell pattern was not driven by the
  threshold.
- Current lead residual model:
  - Residual magnitude remains strongly tied to effort/access: traveling rate
    Spearman 0.7066, stationary rate -0.7066, effort distance mean 0.6141,
    effort distance p90 0.5524, and number of observers 0.4302.
  - Residual direction remains tied to environmental/geographic gradients:
    canopy 0.5484, distance to waterbody 0.4975, observed rate/species per
    checklist 0.4362.
  - This confirms that the current spatial residual is a useful but mixed
    correction: part ecological signal, part effort/access geography.
- Shared effort-bias model:
  - Mean absolute residual corrections shrink substantially, but the largest
    residual cells remain in blocks 79 and 31.
  - Residual magnitude is still associated with traveling rate (0.6415),
    stationary rate (-0.6415), effort distance p90 (0.5319), and effort
    distance mean (0.5282).
  - Residual direction becomes very strongly geographic/environmental:
    elevation 0.8500, distance to coastline 0.8413, and distance to waterbody
    0.7460.
  - Interpretation: the shared bias component reduces some residual amplitude,
    but it does not remove effort/access dependence; it shifts more of the
    residual direction into broad geography.
- Low-rank effort-bias model:
  - Residual corrections shrink compared with the lead residual model, but
    high-magnitude residual cells remain concentrated in block 79 and block 31.
  - Residual magnitude is still tied to traveling rate (0.6623), stationary
    rate (-0.6623), and effort distance mean (0.4811).
  - Residual direction is even more strongly geographic: distance to coastline
    0.8336, elevation 0.8041, distance to waterbody 0.7143.
  - Interpretation: low-rank effort bias smooths the spatial residual but does
    not disentangle suitability from effort/access geography.
- Updated framework implication:
  - The problem is not that the model lacks an effort covariate head. The
    problem is that spatial correction is under-identified: broad geographic
    gradients, observer/access structure, and real habitat gradients are
    entangled in the same residual channel.
  - The next architecture should explicitly regularize or validate spatial
    residual transfer, for example by penalizing residual magnitude where held-
    out block transfer is unstable, adding residual dropout/noise at the cell
    level, or learning separate ecological-neighbor and access-neighbor spatial
    channels with constraints.

Residual regularization/dropout experiment:

- `exp/ebird_spatial_gnn_baseline.py` now supports training-time constraints on
  the spatial residual channel:
  - `--spatial-residual-logit-l2`: L2 penalty on spatial residual logits in
    each training batch. This discourages large residual corrections unless
    they clearly improve the objective.
  - `--spatial-residual-dropout`: training-only dropout on residual logits
    before adding them to the base prediction. This tests whether the model can
    avoid over-relying on cell-specific residual corrections.
  - `--spatial-residual-noise-std`: training-only Gaussian noise on residual
    logits. This tests whether small perturbations improve transfer robustness.
- Evaluation remains deterministic; dropout/noise are only active during
  training.
- This is the next small architecture step because it directly targets the
  under-identified residual channel without adding another free effort/bias
  head.
- Success criteria:
  - Keep micro/macro AUPRC close to the current lead model:
    `spatial_gcn_residual_scaled_sigmoid010_l2_0p01`.
  - Reduce residual magnitude correlations with traveling rate, effort
    distance, observer count, and stationary rate in
    `diagnose_ebird_cell_residual_effort.py`.
  - Improve or at least not worsen block 79 and block 31 block/species
    diagnostics.
- Smoke test passed with capped data; full runs still need to be executed.

Recommended first full residual-regularized run:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_residual_scaled_l2resid_0p001_drop010_noise001 --gnn-mode residual --component-mode joint --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-residual-logit-l2 0.001 --spatial-residual-dropout 0.10 --spatial-residual-noise-std 0.01
```

Initial residual-regularized result:

- `spatial_gcn_residual_scaled_l2resid_0p001_drop010_noise001`
  - micro AUROC 0.8911
  - micro AUPRC 0.5794
  - macro AUROC 0.8439
  - macro AUPRC 0.4122
  - ECE 0.0175
  - max bin error 0.0555
  - species calibration MAE 0.0209
- Interpretation:
  - This kept most of the current lead model's ranking performance
    (`0.5808` micro AUPRC and `0.4126` macro AUPRC for the lead model), so the
    residual channel is not extremely fragile.
  - Calibration degraded sharply: ECE increased from 0.0033 to 0.0175, max-bin
    error from 0.0127 to 0.0555, and species calibration MAE from 0.0118 to
    0.0209.
  - This is not a better model as-is. It is only worth continuing if the cell
    residual/effort diagnostic shows a meaningful reduction in effort-correlated
    residual magnitude or better block transfer.

Residual-regularized cell diagnostic:

- `spatial_gcn_residual_scaled_l2resid_0p001_drop010_noise001` substantially
  reduced residual magnitude:
  - The largest cell-level mean absolute residual corrections dropped from
    about 0.050 in the current lead model to about 0.0245.
  - High-magnitude residual cells are still concentrated mostly in block 79,
    with additional cells from blocks 65 and 31.
- The effort/access signal weakened but did not disappear:
  - `probability_delta_abs_mean` vs traveling rate dropped from Spearman 0.7066
    in the lead model to 0.6097.
  - `probability_delta_abs_mean` vs stationary rate changed from -0.7066 to
    -0.6097.
  - This is a real reduction, but the residual magnitude is still materially
    associated with protocol/effort geography.
- The geographic/environmental signal became more dominant:
  - `probability_delta_mean` vs distance to coastline: 0.7531.
  - `probability_delta_mean` vs distance to waterbody: 0.7050.
  - `probability_delta_mean` vs elevation: 0.6710.
  - `negative_probability_delta_mean` vs distance to coastline: 0.8002.
  - `probability_delta_abs_mean` vs elevation: -0.7526.
- Interpretation:
  - Residual regularization did what it was supposed to do mechanically: it
    made the spatial correction smaller.
  - It did not make the residual clearly more identifiable as ecological
    suitability. Instead, it traded large effort-correlated residuals for
    smaller but still effort-associated residuals that are even more aligned
    with broad coastal/elevation/water gradients.
  - Because calibration degraded sharply while AUPRC stayed close, the first
    regularized run is not a lead model. It is evidence that amplitude control
    alone is not enough.
- Next decision:
  - Run the lighter residual-regularized model once. If the lighter run keeps
    calibration closer to the lead while reducing residual/effort correlation,
    this branch is worth tuning.
  - If the lighter run shows the same pattern, stop residual amplitude
    regularization as a primary approach and move to a more structural
    constraint: separate environmental-neighbor spatial message passing from
    access/checklist-density message passing, then evaluate whether the
    ecological channel transfers better across held-out blocks.

If this over-regularizes, try a lighter version:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_residual_scaled_l2resid_0p0001_drop005 --gnn-mode residual --component-mode joint --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-residual-logit-l2 0.0001 --spatial-residual-dropout 0.05
```

Lighter residual-regularized result:

- `spatial_gcn_residual_scaled_l2resid_0p0001_drop005`
  - micro AUROC 0.8911
  - micro AUPRC 0.5797
  - macro AUROC 0.8439
  - macro AUPRC 0.4118
  - ECE 0.0079
  - max bin error 0.0270
  - species calibration MAE 0.0151
- Compared with the current lead:
  - Ranking is still close: micro AUPRC 0.5797 vs 0.5808, macro AUPRC 0.4118
    vs 0.4126.
  - Calibration is worse than the lead but much less damaged than the stronger
    regularized run: ECE 0.0079 vs 0.0175 for the stronger run and 0.0033 for
    the lead.
- Cell residual diagnostic:
  - Residual magnitude is materially reduced. The largest cell-level mean
    absolute residual correction is 0.0315, compared with about 0.050 in the
    lead and 0.0245 in the stronger regularized run.
  - The effort/protocol association is reduced but remains substantial:
    residual magnitude vs traveling rate Spearman 0.6032, stationary rate
    -0.6032.
  - Residual magnitude remains associated with broad geography/environment:
    elevation -0.6492, distance to coastline -0.6152, distance to waterbody
    -0.5304, canopy -0.4773.
  - Residual direction remains tied to habitat/geography: distance to waterbody
    0.5621, canopy 0.5260, coastline 0.4800.
- Interpretation:
  - The lighter regularization is a better tradeoff than the stronger version.
    It reduces residual amplitude and softens effort correlation while keeping
    most ranking performance and only moderately hurting calibration.
  - It still does not solve identifiability. The residual channel remains
    entangled with effort/protocol and broad geography.
- Decision:
  - Keep this as a useful ablation, not as a new lead architecture.
  - Run effort-strata and block/species diagnostics once to see whether the
    reduced residual amplitude improves the known failure blocks without
    erasing useful graph gains.
  - Unless those diagnostics show a clear block-transfer improvement, pivot to a
    structural split between ecological-neighbor and access/checklist-density
    spatial channels.

Lighter residual-regularized effort/block diagnostics:

- Effort-strata results:
  - High-effort/traveling strata still improve: duration 121+ (+0.0142 micro
    AUPRC), distance (2,5] (+0.0121), distance 5+ (+0.0108), traveling protocol
    (+0.0091), and observers 2 (+0.0086).
  - Low-effort/stationary strata are only mildly positive: stationary (+0.0018),
    zero distance (+0.0018), and duration 1-10 (+0.0018).
  - Block 65 remains strong (+0.0118 micro AUPRC), but block 79 remains a clear
    failure (-0.0120 micro, -0.0107 macro), and block 31 is weak/mixed (+0.0018
    micro, -0.0040 macro).
- Block/species results:
  - Block 65 is again where graph structure helps most: mean delta AUPRC
    +0.0130, median +0.0061, 69 species with gains and 26 losses.
  - Block 79 remains the core local transfer failure: mean delta AUPRC -0.0114,
    median -0.0005, 50 species with losses and 47 with gains. Largest losses
    include House Sparrow (-0.3393), House Finch (-0.2445), European Starling
    (-0.1267), Red-headed Woodpecker (-0.1178 in block 79), Northern
    Mockingbird (-0.1018), Mourning Dove (-0.0934), and Northern Cardinal
    (-0.0813).
  - Block 31 remains mixed/negative: mean delta AUPRC -0.0041, with important
    losses for Eastern Meadowlark (-0.1581), Mallard (-0.0765), Ovenbird
    (-0.0756), Belted Kingfisher (-0.0576), and Pied-billed Grebe (-0.0550).
  - Red-headed Woodpecker remains a severe failure in block 65 (-0.1456) and
    block 79 (-0.1178), reinforcing that reduced residual magnitude does not
    solve species-specific transfer.
- Decision:
  - Stop residual amplitude regularization as the primary route. It reduces
    residual size and preserves aggregate ranking, but it does not fix the
    block/species transfer failures and still leaves effort/geography
    entanglement.
  - Keep `spatial_gcn_residual_scaled_l2resid_0p0001_drop005` as a useful
    ablation showing that the residual can be shrunk without destroying AUPRC,
    but do not treat it as the lead.
  - Proceed to structural separation: build separate spatial message-passing
    channels for ecological/environmental neighborhood structure and
    access/checklist-density neighborhood structure, then constrain the access
    channel so it cannot freely become species-specific suitability.

Separated ecological/access spatial channels:

- `exp/ebird_spatial_gnn_baseline.py` now supports
  `--spatial-channel-mode separated`.
- This is different from the earlier `--component-mode separated` experiment.
  The checklist/species scoring path remains `--component-mode joint`; only the
  spatial message-passing context is split.
- In separated spatial-channel mode:
  - The ecological channel receives cell-level environmental/seasonal features:
    day-of-year sine/cosine, canopy, elevation, distance to waterbody, and
    distance to coastline.
  - The access channel receives spatial/access/effort features: cell centroid,
    train-checklist density, day-of-week sine/cosine, duration, effort distance,
    observer count, and traveling indicator.
  - The species-specific spatial residual head uses only the ecological channel.
  - The access channel can only contribute a shared checklist-level spatial
    bias through a single logit adjustment applied to all species. This prevents
    the access channel from freely becoming a species-specific suitability
    surface.
  - `--spatial-access-bias-l2` optionally penalizes the shared access spatial
    bias logits during training.
- This directly tests the framework question: can a GNN preserve useful
  ecological spatial transfer while keeping observation/access geography in a
  constrained, non-species-specific component?
- Smoke test passed with capped data.
- First full separated-channel run completed:
  - `spatial_gcn_separated_channels_shared_access_l2_0p001`
  - Metrics: micro AUROC 0.8917, micro AUPRC 0.5792, macro AUROC 0.8447,
    macro AUPRC 0.4136, ECE 0.0052, max bin error 0.0216, and species
    calibration MAE 0.0131.
  - Compared with the current lead model
    `spatial_gcn_residual_scaled_sigmoid010_l2_0p01`, this is broadly
    competitive: micro AUPRC is slightly lower, macro AUPRC is slightly higher,
    and calibration is worse but still reasonable.
  - This means the split-channel architecture is viable enough to keep testing,
    but it is not yet a clean replacement for the lead model.
- First separated-channel cell residual/effort diagnostic:
  - Summarized 28 held-out cells with at least 10 checklists.
  - Mean absolute species-specific residual deltas remain largest in cells from
    blocks 31, 65, and 79.
  - Residual magnitude is still associated with traveling protocol, but less
    strongly than the earlier single-channel lead model: Spearman correlation
    between mean absolute residual delta and traveling rate is 0.5161, with
    stationary rate -0.5161.
  - The strongest residual-direction correlations are now ecological/geographic:
    coastline distance, elevation, and waterbody distance dominate the
    correlation table. For example, mean residual delta has Spearman 0.8834
    with coastline distance and 0.8429 with elevation.
  - Interpretation: the split-channel design appears to reduce direct
    effort/protocol entanglement in the species-specific residual, but the
    residual is still strongly geographic. That may be ecological signal, coastal
    sampling structure, or remaining bias aligned with geography. Treat this as
    a promising framework direction, not a solved decomposition.
- Diagnostic update:
  - `exp/diagnose_ebird_cell_residual_effort.py` now reports shared access-bias
    terms for separated-channel runs:
    `access_bias_logit_mean`, `access_bias_logit_abs_mean`,
    `access_probability_delta_mean`, and
    `access_probability_delta_abs_mean`.
  - The updated diagnostic shows partial, not clean, separation:
    - Species-specific residual summaries are now dominated by
      ecological/geographic gradients. The strongest correlations are
      negative residual direction with coastline distance (Spearman 0.9031),
      negative residual direction with elevation (0.8949), mean residual
      direction with coastline distance (0.8834), mean residual direction with
      elevation (0.8429), and mean residual direction with waterbody distance
      (0.7603).
    - Residual magnitude still correlates with protocol geography, but less
      strongly than the earlier lead model: mean absolute residual delta vs
      stationary rate is -0.5161 and vs traveling rate is 0.5161.
    - The shared access-bias probability delta does appear in the effort/access
      correlation table, especially mean absolute access delta vs observer
      count (0.5320). It also correlates with observed rate and species per
      checklist (about +/-0.527), meaning it is not a purely effort-only term.
  - Interpretation: the split-channel architecture is moving in the intended
    direction. It pushes the species-specific residual away from direct
    protocol/effort dominance and toward ecological/geographic gradients, while
    a constrained shared access term picks up some observer/access structure.
    However, this is not yet a fully identifiable bias/suitability
    decomposition because access, ecology, and observed richness remain
    geographically aligned in the NC data.
  - Decision: continue validating this branch, but judge it by held-out
    strata/block behavior rather than by the cell-correlation table alone.
- Separated-channel effort-strata diagnostics:
  - The high-effort pattern remains, but the separated-channel model is more
    balanced across effort strata than the earlier lead residual model.
  - Largest micro-AUPRC gains are still in higher-effort contexts: distance
    `5+` km (+0.0126), duration `121+` minutes (+0.0116), distance `(2,5]`
    km (+0.0109), traveling checklists (+0.0089), 2 observers (+0.0085), and
    3+ observers (+0.0083).
  - Importantly, low-distance strata are no longer losses: distance `(0,0.5]`
    km improves by +0.0064 and zero-distance/stationary checklists are nearly
    neutral rather than strongly negative (+0.0005 micro AUPRC for both).
  - Calibration changes are mixed but mostly modest. Some high-effort strata
    gain ranking with worse ECE, while several low-effort/low-distance strata
    improve ECE.
  - Interpretation: the separated-channel model preserves the useful graph
    signal in high-effort contexts while reducing the earlier weakness in
    short-distance/near-stationary strata. That is consistent with the goal of
    constraining observer/access effects rather than letting them dominate the
    species-specific spatial residual.
- Separated-channel block/species diagnostics:
  - Block 65 remains the strongest transfer success: mean delta AUPRC +0.0095,
    median +0.0056, 70 species with gains and 25 with losses.
  - Block 31 is close to neutral/slightly positive: mean delta AUPRC +0.0018,
    median +0.0011, 53 species with gains and 42 with losses.
  - Block 79 remains the core failure: mean delta AUPRC -0.0079, median
    -0.0009, 51 species with losses and 46 with gains.
  - Largest block/species gains include Black-and-white Warbler in block 65
    (+0.1128), Bald Eagle in block 65 (+0.0851), Eastern Towhee in block 65
    (+0.0677), Red-tailed Hawk in block 79 (+0.0657), and White-eyed Vireo in
    block 65 (+0.0611).
  - Largest losses remain concentrated in block 79 and selected species:
    House Finch in block 79 (-0.2573), House Sparrow in block 79 (-0.1605),
    Bufflehead in block 65 (-0.1109), Red-headed Woodpecker in block 65
    (-0.1081), Northern Cardinal in block 79 (-0.0976), Mourning Dove in block
    79 (-0.0968), and European Starling in block 79 (-0.0844).
  - Interpretation: separated spatial channels improve effort-stratum behavior
    but do not eliminate species/block transfer failures. This supports keeping
    the separated-channel architecture as the current framework direction, while
    focusing next on why block 79 and a few species are harmed.
- Decision after separated-channel diagnostics:
  - Treat `spatial_gcn_separated_channels_shared_access_l2_0p001` as the best
    current framework branch for bias/suitability separation, but not as a
    final model.
  - Do not add more architecture complexity yet. First inspect maps for the
    major separated-channel losses and compare them against the earlier lead
    residual maps.
  - If the maps show broad suppression of true positives, the next model change
    should be a softer ecological residual gate or a residual prior that limits
    negative correction on sparse/localized species.
  - If the maps show localized coastal/block artifacts, the next change should
    target spatial cell graph construction or held-out block design, not the
    species scoring head.
- Separated-channel residual-map observations:
  - The species residual maps are still visually block/cell-shaped corrections,
    not smooth fine-scale ecological surfaces. This is expected from the
    current spatial-cell graph, but it means interpretation should focus on
    transfer behavior between held-out spatial blocks rather than on continuous
    range-map realism.
  - Several losses are broad negative corrections rather than subtle local
    ranking changes:
    - Mallard is strongly suppressed overall: mean probability drops from
      0.2025 to 0.1318, with positive checklist mean delta -0.1070 and
      all-checklist mean delta -0.0707.
    - Bufflehead is suppressed especially on positives: mean probability drops
      from 0.0517 to 0.0315, with positive mean delta -0.0901.
    - House Sparrow, Red-headed Woodpecker, European Starling, and Belted
      Kingfisher also receive negative mean corrections, including negative
      corrections on positive checklists.
  - Some apparent block/species losses are more nuanced:
    - House Finch has a small overall negative mean delta (-0.0186) and similar
      positive/negative deltas, so its large block-79 AUPRC loss is probably a
      local ranking problem rather than just statewide suppression.
    - Northern Cardinal and Mourning Dove receive positive overall corrections
      but still show block-79 losses, implying that over-boosting negatives in
      the wrong block can hurt ranking even when positives are boosted on
      average.
    - Eastern Towhee is spatially mixed: western positives are boosted while
      central/coastal positive clusters are suppressed.
    - Black-and-white Warbler has almost no mean residual change, consistent
      with a gain that likely comes from small ranking adjustments rather than
      broad probability movement.
  - Interpretation: the separated-channel residual is less effort-dominated
    than the original residual, but species-specific corrections can still be
    too blunt at the held-out-block scale. The next diagnostic should inspect
    the shared access-bias map side-by-side with the species residual. If the
    access-bias panel is absorbing broad observer/access geography, then the
    remaining species residual can be refined with a softer gate. If the access
    panel is weak or environmentally patterned, the separation is still not
    doing enough.
- Diagnostic update:
  - `exp/plot_ebird_spatial_gnn_residual_maps.py` now adds a third panel for
    separated-channel runs: shared access probability delta
    (`base - no-access probability`). Older single-channel runs still produce
    the original residual/positives two-panel maps.
- Shared access-panel result:
  - The shared access probability delta is present but small relative to the
    species-specific residual. It is also negative on average for every mapped
    species, which means the learned access term is mostly acting as a
    species-shared downward adjustment rather than a rich access surface.
  - Examples:
    - Mallard: species residual mean delta -0.0707, access mean delta -0.0119.
    - Bufflehead: species residual mean delta -0.0202, access mean delta
      -0.0041, but positive access delta is -0.0224.
    - House Finch: species residual mean delta -0.0186, access mean delta
      -0.0141.
    - Northern Cardinal: species residual mean delta +0.0656, access mean
      delta -0.0154.
  - Interpretation: this confirms the conceptual issue. The access channel is
    not sufficiently identified by detection BCE alone. It can become a generic
    offset, while the species residual still carries most block-scale
    correction. The next model should give the access channel an explicit
    observation-process target instead of relying only on all-species detection
    loss.
- Access-density auxiliary experiment:
  - `exp/ebird_spatial_gnn_baseline.py` now supports
    `--access-density-loss-weight`.
  - This is only valid with `--spatial-channel-mode separated`.
  - When enabled, the access-cell embedding gets an auxiliary MSE loss to
    predict standardized log train-checklist density for each spatial cell.
    This makes the access channel explicitly encode an observation-effort
    surface, while the species-specific residual still uses the ecological cell
    channel.
  - This does not make the access component a true detection probability or
    true sampling process by itself. It is a practical identifiability aid:
    access geography now has a direct target, rather than being learned only
    through detection BCE.
  - Smoke test passed with capped data.
  - First full run completed:
    - `spatial_gcn_separated_channels_access_density_w0p01`
    - Metrics: micro AUROC 0.8915, micro AUPRC 0.5799, macro AUROC 0.8447,
      macro AUPRC 0.4147, ECE 0.0057, max bin error 0.0189, and species
      calibration MAE 0.0135.
    - Compared with the previous separated-channel run
      `spatial_gcn_separated_channels_shared_access_l2_0p001`, this is a small
      ranking improvement: micro AUPRC rises from 0.5792 to 0.5799 and macro
      AUPRC rises from 0.4136 to 0.4147. Calibration is slightly worse by ECE
      and species calibration MAE, but max bin error is slightly better.
    - Compared with the current lead residual model
      `spatial_gcn_residual_scaled_sigmoid010_l2_0p01`, aggregate performance
      is still very close: slightly lower micro AUPRC, slightly higher macro
      AUPRC, and somewhat worse calibration.
    - Interpretation: the auxiliary access-density target did not disrupt the
      detector and may modestly help species-level ranking. The key question is
      now decomposition, not aggregate metrics: did the access channel become a
      more meaningful access/effort surface, and did the species residual become
      less responsible for broad block-scale corrections?
  - Access-density cell diagnostic:
    - The largest residual magnitudes remain concentrated in the same problem
      cells/blocks, especially block 79 and block 31.
    - The species residual remains geographically structured. The strongest
      species-residual correlations include negative residual direction with
      elevation (Spearman 0.8697), negative residual direction with coastline
      distance (0.8533), mean residual direction with waterbody distance
      (0.8270), mean residual direction with coastline distance (0.8008), and
      mean residual direction with elevation (0.7674).
    - The access channel became more active, but it mostly learned broad
      geographic/access gradients rather than a clearly effort-only component:
      access probability delta correlates with elevation (0.8741), coastline
      distance (0.8462), and waterbody distance (0.7165); access-bias logit
      mean also correlates with elevation (0.8560), coastline distance
      (0.8369), and waterbody distance (0.6957).
    - This is not surprising because train-checklist density is itself
      geographically structured in NC. Explicitly predicting checklist density
      teaches the access channel an observation geography surface, but that
      surface is still entangled with coast/elevation/water gradients.
    - Interpretation: the auxiliary target is useful but insufficient. It gives
      the access channel a real target and does not hurt aggregate performance,
      but it does not cleanly separate access from ecological geography. The
      next criterion is whether it improves held-out effort strata and the
      block/species failures. If it does not, the next architecture change
      should use richer access-process supervision, not just checklist density.
  - Access-density effort-strata diagnostics:
    - The high-effort gains remain and are slightly stronger than the previous
      separated-channel run in several strata: duration `121+` minutes improves
      by +0.0132 micro AUPRC, distance `(2,5]` km by +0.0118, traveling
      checklists by +0.0099, and 2-observer checklists by +0.0091.
    - Low-distance strata also remain positive: distance `(0,0.5]` km improves
      by +0.0086, distance `(0.5,2]` by +0.0076, zero-distance checklists by
      +0.0008, and stationary checklists by +0.0007.
    - The major problem is spatial block 79. Its micro-AUPRC delta worsens to
      -0.0184, compared with -0.0047 for the previous separated-channel run.
      Block 31 improves to +0.0046 and block 65 improves to +0.0116.
    - Interpretation: access-density supervision improves effort-stratum
      balance and the two better held-out blocks, but it makes the hardest
      held-out block substantially worse in aggregate. This suggests the
      auxiliary target is helping the access channel encode broad effort
      geography, but may also be amplifying a block-79 mismatch.
  - Access-density block/species diagnostics:
    - Block 65 remains strong: mean delta AUPRC +0.0099, median +0.0080, 69
      species with gains and 26 with losses.
    - Block 31 improves relative to the previous separated-channel run: mean
      delta AUPRC +0.0029, median +0.0021, 55 species with gains and 40 with
      losses.
    - Block 79 remains negative but the species-level mean is less bad than the
      previous separated-channel run: mean delta AUPRC -0.0053 versus -0.0079.
      However, micro-AUPRC in block 79 is much worse, so the losses are likely
      concentrated in influential/common species or ranking structure.
    - Some major block-79 losses improved:
      - House Sparrow improves from -0.1605 to -0.0718.
      - Northern Cardinal improves from -0.0976 to -0.0729.
      - Mourning Dove improves from -0.0968 to -0.0606.
      - House Finch improves slightly from -0.2573 to -0.2409.
    - Some losses worsened or persisted:
      - European Starling worsens from -0.0844 to -0.1002.
      - Bufflehead in block 65 worsens slightly from -0.1109 to -0.1165.
      - Red-headed Woodpecker in block 65 worsens slightly from -0.1081 to
        -0.1142.
    - Interpretation: access-density supervision redistributes the failures
      rather than solving them. It improves several species-specific failures
      and the median block behavior, but the harder block-79 aggregate ranking
      problem remains.
  - Decision after access-density diagnostics:
    - Keep `spatial_gcn_separated_channels_access_density_w0p01` as a serious
      branch because it modestly improves aggregate AUPRC and improves some
      effort/block behavior.
    - Do not declare it the lead framework model yet because block 79
      aggregate micro-AUPRC worsened sharply.
    - Next inspect residual/access maps for the access-density run. The key
      question is whether the access-density target moved broad block-level
      correction into the shared access panel or whether species residuals still
      carry most of the harmful block-79 correction.
  - Access-density residual/access map result:
    - The shared access panel remains small relative to the species-specific
      residual and is still mostly a downward adjustment across the mapped
      species.
    - The auxiliary density target did not move the broad correction into the
      shared access panel. For several species, the species residual became
      more negative than in the previous separated-channel run:
      - House Finch mean residual delta moved from -0.0186 to -0.0358.
      - House Sparrow moved from -0.0202 to -0.0269.
      - Red-headed Woodpecker moved from -0.0161 to -0.0227.
      - European Starling moved from -0.0156 to -0.0203.
    - Some broad suppressions improved but remained species-residual driven:
      - Mallard mean residual delta improved from -0.0707 to -0.0606.
      - Bufflehead improved from -0.0202 to -0.0187.
    - The access delta stayed small:
      - House Finch access delta -0.0126 versus species residual -0.0358.
      - Mallard access delta -0.0091 versus species residual -0.0606.
      - Bufflehead access delta -0.0030 versus species residual -0.0187.
      - Northern Cardinal access delta -0.0119 while species residual is
        strongly positive (+0.0647).
    - Interpretation: checklist-density supervision alone is not enough. It
      creates a more active access channel but does not make that channel absorb
      the problematic block-scale correction. The species residual still carries
      most of the model's broad spatial adjustment.
  - Decision after residual/access maps:
    - Treat `spatial_gcn_separated_channels_access_density_w0p01` as an
      informative ablation, not the lead.
    - The better framework direction is still separated ecological/access
      channels, but the access component needs richer observation-process
      structure than checklist density alone.
    - Avoid another one-parameter density-loss sweep unless needed as an
      appendix. The next model change should either:
      - explicitly predict multiple access/effort summaries from the access
        channel, such as checklist density, traveling rate, mean duration,
        effort distance, observer count, and locality/observer turnover; or
      - use a two-stage access encoder trained/frozen on effort/access targets,
        then let the species detector use that fixed access representation.
    - The two-stage route is preferable for framework clarity because it makes
      the observation-process representation less able to drift into a generic
      species-loss correction.

Two-stage access encoder:

- `exp/train_ebird_access_encoder.py` trains the access/effort spatial-cell
  encoder as a separate first-stage observation-process model.
- This is different from `--access-density-loss-weight`, which added one
  auxiliary density target while still training the whole species detector
  jointly.
- The two-stage access encoder predicts multiple train-only cell-level
  access/effort summaries:
  - log train checklist count
  - traveling rate
  - stationary rate
  - mean log duration
  - mean log effort distance
  - mean log observer count
  - log unique observers
  - log unique localities
  - observers per checklist
  - localities per checklist
- The model uses the existing spatial-cell graph and access-channel inputs
  (centroid, train-checklist density, day-of-week, duration, distance,
  observers, traveling indicator). It saves:
  - cell embeddings
  - standardized predictions
  - cell target table
  - per-target train/validation metrics
  - model state and JSON metadata
- This is intended as an identifiability step, not an endpoint. If the access
  encoder learns useful held-out access summaries, the next species model can
  use its frozen access embeddings as an observation-process representation
  while keeping species-specific residuals on the ecological channel.
- Smoke test passed with a two-epoch capped run.
- First full access-encoder run completed:
  - `access_gcn_h64_l2_z64`
  - Validation MSE improved from 1.204 at epoch 1 to about 0.576 by epoch 500,
    with the best validation loss around epoch 200. The slight later drift is
    mild and acceptable for this first representation test.
  - Validation Pearson correlations are nontrivial for most access summaries:
    observer count 0.5549, locality per checklist 0.4742, checklist density
    0.4576, observer per checklist 0.4523, unique localities 0.3877, unique
    observers 0.3745, effort distance 0.3543, duration 0.3082, traveling rate
    0.1998, and stationary rate 0.1990.
  - Interpretation: the access encoder is not perfect, but it learned a real
    observation-process representation. It is good enough to test as a frozen
    access channel in the species detector.
- `exp/ebird_spatial_gnn_baseline.py` now supports
  `--frozen-access-embeddings`. In separated spatial-channel mode, this replaces
  the learned access-cell channel with a pretrained access embedding while the
  ecological cell channel and species residual remain trainable.
- Diagnostics and residual maps now reload frozen-access runs correctly.
- Frozen-access species-model smoke test passed with capped data.

First full access-encoder command:

```
python exp/train_ebird_access_encoder.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name access_gcn_h64_l2_z64 --epochs 500 --hidden-dim 64 --layers 2 --embedding-dim 64 --dropout 0.10 --spatial-grid-size-m 25000
```

Readout:

- Look first at validation Pearson and MSE in
  `data/ebird/graph_top100_spatial_10x10/access_encoder/access_gcn_h64_l2_z64_target_metrics.csv`.
- Good enough for the next step means the encoder learns checklist density and
  at least several effort/access summaries with nontrivial validation
  correlation. It does not need to predict every target perfectly.
- If validation correlations are near zero across most targets, the access
  summaries are either too sparse/noisy at this cell scale or the access
  encoder needs different inputs/cell resolution before being used in the
  species detector.

First frozen-access species-model command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_frozen_access_h64_l2_z64 --gnn-mode residual --component-mode joint --spatial-channel-mode separated --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-access-bias-l2 0.001 --frozen-access-embeddings data/ebird/graph_top100_spatial_10x10/access_encoder/access_gcn_h64_l2_z64_cell_embeddings.npy
```

Readout goal:

- If frozen access improves or preserves aggregate AUPRC while reducing
  residual/access entanglement, this becomes the preferred framework direction.
- If aggregate performance drops modestly but block/species transfer improves,
  it may still be worthwhile because the framework goal is better
  bias/suitability separation, not only maximizing NC/top-100 AUPRC.
- If both performance and diagnostics worsen, the access encoder should remain
  an ablation and the next step should revisit access targets/cell resolution.

First frozen-access species-model result:

- `spatial_gcn_frozen_access_h64_l2_z64`
  - micro AUROC 0.8919
  - micro AUPRC 0.5800
  - macro AUROC 0.8450
  - macro AUPRC 0.4151
  - ECE 0.0056
  - max bin error 0.0193
  - species calibration MAE 0.0133
- This is the best separated-channel branch so far by macro AUPRC, and it
  matches or slightly improves the access-density branch on micro AUPRC.
- Compared with the current lead residual model
  `spatial_gcn_residual_scaled_sigmoid010_l2_0p01`, frozen access is still a
  little lower on micro AUPRC but higher on macro AUPRC. Calibration remains
  worse than the lead but acceptable.
- Interpretation: using a pretrained/frozen access representation did not hurt
  the species detector. That is encouraging for the framework goal because it
  means access-process structure can be separated more explicitly without
  sacrificing much aggregate detection performance. The next question is
  whether the diagnostics show better decomposition and block/species transfer.

Frozen-access diagnostics:

```
python exp/diagnose_ebird_cell_residual_effort.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_frozen_access_h64_l2_z64 --min-cell-checklists 10
```

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_frozen_access_h64_l2_z64
```

```
python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_frozen_access_h64_l2_z64
```

Frozen-access cell residual/effort diagnostic:

- The frozen-access branch still does not produce a clean access/ecology
  decomposition at the cell-correlation level. Species residual direction
  remains strongly tied to broad geography:
  - negative residual direction vs coastline distance: Spearman 0.8402
  - negative residual direction vs elevation: 0.8342
  - mean residual direction vs waterbody distance: 0.8221
  - negative residual direction vs waterbody distance: 0.8183
  - mean residual direction vs coastline distance: 0.7750
- The access component is still geographically structured, but less extreme
  than the joint access-density branch:
  - access probability delta vs elevation: 0.6814
  - access probability delta vs coastline distance: 0.6798
  - access probability delta vs waterbody distance: 0.5720
  - access-bias logit mean vs coastline distance: 0.6070
  - access-bias logit mean vs elevation: 0.5868
- Residual magnitudes are somewhat moderated compared with the joint
  access-density branch. The largest mean absolute residual delta is 0.0344,
  compared with 0.0388 for the joint access-density model.
- Interpretation: freezing the pretrained access representation improves
  identifiability relative to the joint access-density auxiliary model, but it
  does not solve the fundamental geography/access/ecology confounding. The
  practical question is therefore whether frozen access improves held-out
  transfer enough to justify carrying this branch forward.

Frozen-access effort-strata diagnostic:

- The effort-strata pattern is stronger and more balanced than the joint
  access-density branch:
  - duration `121+`: +0.0134 micro AUPRC
  - distance `(2,5]`: +0.0130
  - 3+ observers: +0.0116
  - distance `5+`: +0.0106
  - traveling: +0.0095
  - 2 observers: +0.0098
- Low-effort/low-distance strata remain positive rather than harmful:
  stationary +0.0019, zero-distance +0.0019, duration `1-10` +0.0045,
  distance `(0,0.5]` +0.0065.
- Block behavior is much better than the joint access-density branch:
  - block 65: +0.0101 micro AUPRC
  - block 31: +0.0036
  - block 79: -0.0057
- Compared with the joint access-density branch, block 79 improves sharply
  from -0.0184 to -0.0057. Compared with the previous separated-channel model
  without frozen access, block 79 is slightly worse than -0.0047 but still in
  the same range.
- Interpretation: frozen access is now the most credible separated-access
  framework branch. It preserves the effort-strata improvements, avoids the
  block-79 collapse from joint access-density supervision, and has the cleanest
  conceptual separation so far. It still needs block/species diagnostics before
  being treated as the preferred branch.

Frozen-access block/species diagnostic:

- Block 65 remains the strongest success:
  - mean delta AUPRC +0.0102
  - median delta AUPRC +0.0059
  - 67 species with gains and 28 with losses
  - largest gains include Black-and-white Warbler (+0.0927), White-eyed Vireo
    (+0.0630), Bald Eagle (+0.0603), Eastern Towhee (+0.0569), and Killdeer
    (+0.0536).
- Block 31 is modestly positive and better balanced than the original
  separated-channel run:
  - mean delta AUPRC +0.0025
  - median delta AUPRC +0.0018
  - 59 species with gains and 36 with losses
  - the main reported loss is Mallard (-0.0470), which is smaller than the
    earlier block-31 Mallard loss under the non-frozen separated-channel run.
- Block 79 remains the core failure:
  - mean delta AUPRC -0.0079
  - median delta AUPRC -0.0010
  - 53 species with losses and 44 with gains
  - largest losses include House Finch (-0.2491), House Sparrow (-0.2183),
    European Starling (-0.1115), Mourning Dove (-0.0797), Northern Mockingbird
    (-0.0733), Northern Cardinal (-0.0728), and Yellow-throated Warbler
    (-0.0665).
- Compared with the previous separated-channel run without frozen access:
  - block 65 is slightly better on mean AUPRC (+0.0102 vs +0.0095).
  - block 31 is better (+0.0025 vs +0.0018), with fewer species losses.
  - block 79 is essentially unchanged on mean AUPRC (-0.0079 in both), though
    individual species losses shift.
- Compared with the joint access-density auxiliary branch:
  - frozen access avoids the block-79 aggregate micro-AUPRC collapse seen in
    effort-strata diagnostics.
  - block/species mean in block 79 is worse than the joint branch
    (-0.0079 vs -0.0053), but the joint branch had much worse block-79 micro
    behavior, so it is not preferable overall.
- Decision:
  - Treat `spatial_gcn_frozen_access_h64_l2_z64` as the preferred
    separated-access framework branch so far.
  - Do not claim it solves the central problem. The recurring block-79 failures
    show that broad held-out geographic transfer is still the limiting issue.
  - The next diagnostic should move from model architecture to validation
    geometry: characterize block 79 versus blocks 31 and 65 in terms of
    geography, effort/access summaries, ecological covariates, species
    composition, and access-encoder targets. If block 79 is an outlier in those
    summaries, the failure is likely a spatial-transfer/support problem rather
    than just an architecture problem.

Held-out spatial block profile:

- `exp/diagnose_ebird_spatial_blocks.py` profiles train/test spatial blocks by
  effort/access, ecological covariates, species composition, and optional
  access-encoder target/prediction summaries.
- Initial command:

```
python exp/diagnose_ebird_spatial_blocks.py --graph-dir data/ebird/graph_top100_spatial_10x10 --access-run-name access_gcn_h64_l2_z64
```

- Outputs are written to
  `data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/diagnostics/block_profile`:
  - `block_profile_summary.csv`
  - `block_species_prevalence.csv`
  - `test_block_species_distance.csv`
  - `access_encoder_target_metrics.csv`
  - `block_profile_metadata.json`
- Initial block-profile result:
  - Block 79 is a qualitatively different held-out geography from blocks 31 and
    65. It is coastal/island/ocean-adjacent rather than inland or mixed.
  - Block 79 has very low elevation (mean 0.93), extremely low distance to
    waterbody (mean 1.36 km), and extremely low distance to coastline
    (mean 1.03 km). By contrast, block 31 is high-elevation/inland
    (mean elevation 767.24, coastline distance 497.69 km) and block 65 is
    lower-elevation but still much farther from the coast (mean elevation
    135.29, coastline distance 230.27 km).
  - Block 79 is also effort-distinct: traveling rate 0.7678, mean effort
    distance 1.388 km, and mean observers 1.7396. Its z-scores versus train
    blocks include high observer count (+2.77), high traveling rate (+2.13),
    low stationary rate (-2.13), high x/eastern location (+1.61), and high
    effort distance (+1.59).
  - Block 79 has lower species-per-checklist mean (10.11) than block 31
    (11.77) and block 65 (15.54), but its species composition is more
    distinctive. Its nearest-train-block species-prevalence L2 distance is
    0.6380, and its distance to the train mean is 1.4735, much larger than
    block 31 (0.6365 to train mean) or block 65 (0.7432).
  - Top block-79 species are coastal/water-associated: Double-crested
    Cormorant, Red-winged Blackbird, Boat-tailed Grackle, Laughing Gull, and
    American Herring Gull. Blocks 31 and 65 are dominated by more inland/common
    woodland or generalist species such as American Crow, Carolina Chickadee,
    Tufted Titmouse, Northern Cardinal, and Carolina Wren.
- Interpretation:
  - The persistent block-79 failure is not just an architecture bug. It is a
    spatial-transfer/support problem: block 79 is a distinct coastal assemblage
    with effort/access and ecological values that differ materially from the
    training block distribution.
  - This explains why architecture changes redistribute errors but do not solve
    the block-79 issue. The model is being asked to transfer into a held-out
    coastal regime that has limited analogs in training.
  - Next model changes should therefore focus on better support/graph structure
    for coastal or environmentally similar cells, not only stronger residual
    regularization. Candidate directions:
    - add environmental-neighbor edges so coastal/water-associated cells can
      borrow information from ecologically similar cells even if not adjacent;
    - use block-aware validation summaries as a required diagnostic for every
      future architecture;
    - consider split designs that explicitly hold out multiple coastal and
      inland blocks so coastal transfer is tested with better replication.
  - Current validation decision:
    - Keep the current 10x10 spatial holdout as a stress test for now. Block 79
      is difficult because it is coastal and partly out-of-support, but that is
      useful for diagnosing whether the framework can transfer beyond the most
      common inland observer geography.
    - Do not tune the split away from block 79 yet. First test whether graph
      structure can improve transfer by connecting spatial cells that are
      environmentally similar, even when they are not adjacent.
    - After this architecture test, add a second split family with explicit
      coastal/inland and effort stratification. That future split should be used
      as a balanced benchmark, while the current block-79 holdout remains a hard
      coastal stress test.
  - Next implementation step:
    - Add optional spatial-cell edge modes:
      - `spatial`: existing queen adjacency, used as the default and for all
        existing runs.
      - `environmental`: k-nearest cells using standardized ecological cell
        summaries.
      - `hybrid`: union of queen spatial adjacency and environmental nearest
        neighbors.
    - Start with `hybrid` edges and the frozen access encoder. This keeps local
      spatial smoothing, preserves the explicit access/bias channel, and gives
      coastal/water-associated cells a path to borrow signal from similar cells
      elsewhere in the state.

Hybrid environmental-neighbor edge command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_frozen_access_env_edges_k6 --gnn-mode residual --component-mode joint --spatial-channel-mode separated --cell-edge-mode hybrid --environmental-neighbors 6 --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-access-bias-l2 0.001 --frozen-access-embeddings data/ebird/graph_top100_spatial_10x10/access_encoder/access_gcn_h64_l2_z64_cell_embeddings.npy
```

After running it, repeat the same diagnostics:

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_frozen_access_env_edges_k6
```

```
python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_frozen_access_env_edges_k6
```

```
python exp/diagnose_ebird_cell_residual_effort.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_frozen_access_env_edges_k6 --min-cell-checklists 10
```

Hybrid environmental-neighbor edge result:

- `spatial_gcn_frozen_access_env_edges_k6`
  - micro AUROC 0.8920
  - micro AUPRC 0.5792
  - macro AUROC 0.8449
  - macro AUPRC 0.4152
  - ECE 0.0048
  - max bin error 0.0202
  - species calibration MAE 0.0125
- Compared with the spatial-only frozen-access model
  `spatial_gcn_frozen_access_h64_l2_z64`, hybrid environmental-neighbor edges
  do not materially improve aggregate ranking:
  - micro AUPRC moves from 0.5800 to 0.5792.
  - macro AUPRC moves from 0.4151 to 0.4152.
  - micro AUROC moves from 0.8919 to 0.8920.
  - macro AUROC moves from 0.8450 to 0.8449.
- Calibration is modestly better:
  - ECE improves from 0.0056 to 0.0048.
  - species calibration MAE improves from 0.0133 to 0.0125.
  - max bin error is slightly worse, from 0.0193 to 0.0202.
- Interpretation:
  - Adding environmental-neighbor edges is not a broad aggregate win at k=6,
    but it also does not destabilize the frozen-access framework. The main
    reason to continue evaluating it is whether it improves block-79/coastal
    transfer or species-specific behavior in ways that aggregate metrics hide.
  - If block/species diagnostics do not improve, the next framework direction
    should probably shift from changing cell adjacency to stronger ecological
    representation: richer environmental covariates, explicit coastal/water
    context, species traits/groups, or a larger geographic training domain.

Hybrid environmental-neighbor diagnostics:

- Effort-strata diagnostics:
  - The high-effort pattern remains. Largest micro-AUPRC gains are still in
    long-duration, higher-distance, multi-observer, and traveling strata:
    duration `121+` minutes (+0.0135), distance `(2,5]` km (+0.0123),
    distance `5+` km (+0.0118), 3+ observers (+0.0106), and traveling
    checklists (+0.0094).
  - Block 79 improves materially relative to the frozen-access spatial-only
    model: block-79 micro-AUPRC delta improves from about -0.0057 to -0.0020.
    This is the main positive signal from hybrid environmental-neighbor edges.
  - Stationary/zero-distance checklists become the weakest effort strata:
    stationary (-0.0009) and zero distance (-0.0008). This suggests the hybrid
    graph is helping transfer among higher-effort/coastal cells more than it is
    helping stationary low-distance checklists.
- Block/species diagnostics:
  - Block 79 improves from the prior frozen-access block/species mean AUPRC
    delta of about -0.0079 to -0.0046. It also shifts from more losses than
    gains to 43 species losses and 54 gains.
  - Block 65 remains clearly positive: mean block/species AUPRC delta +0.0113,
    with 63 gains and 32 losses.
  - Block 31 is essentially neutral: mean block/species AUPRC delta +0.0008,
    with 54 gains and 41 losses.
  - The largest block-79 gains are water/coastal-associated species, including
    Pied-billed Grebe (+0.0753), Canada Goose (+0.0634), Ring-billed Gull
    (+0.0523), Bufflehead (+0.0521), and Yellow-rumped Warbler (+0.0512).
    This is consistent with the purpose of environmental-neighbor edges.
  - The largest block-79 losses remain common/generalist or human-associated
    species: House Finch (-0.2905), House Sparrow (-0.2808), European Starling
    (-0.1361), Northern Cardinal (-0.0799), Mourning Dove (-0.0764), and
    Northern Mockingbird (-0.0747). This means the hybrid graph helps some
    coastal/water assemblage transfer while still misranking several common
    species in the coastal holdout.
- Cell residual/effort diagnostics:
  - The residual remains strongly aligned with broad geography: Spearman
    correlations are high with elevation (+0.9349 for mean probability delta)
    and distance to coastline (+0.8894). Access probability deltas and access
    bias logits also correlate with coastline/elevation.
  - This is not necessarily wrong for a coastal transfer test, but it means the
    framework still has not cleanly separated access geography from ecological
    geography. Hybrid edges improve block-79 transfer, but they do not solve
    the identifiability problem.
- Decision:
  - Keep `spatial_gcn_frozen_access_env_edges_k6` as a useful comparison branch,
    not as a clear replacement for the spatial-only frozen-access branch.
  - The next check should test whether the hybrid-edge benefit is robust to the
    environmental-neighbor count. Try smaller and larger k values before adding
    new architecture. If k changes only trade common-species losses against
    water/coastal gains, then the next framework step should be richer
    ecological/coastal covariates or a broader training domain.

Hybrid environmental-neighbor k-sensitivity:

- `spatial_gcn_frozen_access_env_edges_k3`
  - micro AUROC 0.8921
  - micro AUPRC 0.5797
  - macro AUROC 0.8451
  - macro AUPRC 0.4156
  - ECE 0.0062
  - max bin error 0.0212
  - species calibration MAE 0.0131
- `spatial_gcn_frozen_access_env_edges_k12`
  - micro AUROC 0.8916
  - micro AUPRC 0.5780
  - macro AUROC 0.8444
  - macro AUPRC 0.4123
  - ECE 0.0037
  - max bin error 0.0190
  - species calibration MAE 0.0117
- Interpretation:
  - Smaller k is better for aggregate ranking. k=3 is the best hybrid-edge
    ranking result so far, though it is still only close to the spatial-only
    frozen-access branch rather than clearly better.
  - Larger k is better for calibration but worse for ranking. k=12 likely
    smooths too broadly across environmental neighbors, which dampens
    species-specific ranking signal while stabilizing probabilities.
  - The ranking/calibration tradeoff is coherent: adding more environmental
    neighbors increases smoothing. The useful range appears to be low-to-moderate
    k, with k=3 and k=6 worth diagnostics and k=12 mainly a calibration
    reference.
- Decision:
  - Use `spatial_gcn_frozen_access_env_edges_k3` as the next primary
    k-sensitivity diagnostic run because it gives the best aggregate ranking
    among hybrid-edge variants.
  - Compare k=3 against k=6 specifically on block 79 and common-species losses.
    If k=3 preserves the block-79 improvement while reducing common-species
    losses, it becomes the preferred hybrid branch. If k=6 is better for block
    79/coastal species, keep k=6 as a targeted coastal-transfer branch.

k=3 hybrid environmental-neighbor diagnostics:

- Effort-strata diagnostics:
  - The same high-effort pattern remains: strongest gains are duration `121+`
    minutes (+0.0133 micro AUPRC), distance `(2,5]` km (+0.0126), distance
    `5+` km (+0.0113), 3+ observers (+0.0106), and traveling checklists
    (+0.0098).
  - Block 79 is still the weakest spatial block: -0.0031 micro AUPRC and
    -0.0060 macro AUPRC. This is better than the spatial-only frozen-access
    branch, but worse than the k=6 hybrid result for block 79 (-0.0020).
  - Stationary/zero-distance strata are essentially neutral rather than
    meaningfully improved. This again suggests the environmental-neighbor graph
    mostly helps higher-effort/traveling contexts.
- Block/species diagnostics:
  - Block 79 is worse under k=3 than k=6 on mean block/species AUPRC:
    -0.0077 for k=3 versus -0.0046 for k=6.
  - Block 79 also shifts back to more losses than gains: 50 species losses and
    47 gains for k=3, versus 43 losses and 54 gains for k=6.
  - Block 65 remains positive under k=3 (+0.0106 mean AUPRC, 66 gains and
    29 losses), and block 31 remains near neutral (+0.0020).
  - The same common/coastal-block failures persist: House Finch (-0.2899),
    House Sparrow (-0.2552), European Starling (-0.1371), Mourning Dove
    (-0.0896), Northern Cardinal (-0.0815), and Northern Mockingbird
    (-0.0760) in block 79.
  - k=3 does not solve the common-species loss problem. It slightly improves
    House Sparrow relative to k=6 but leaves House Finch and European Starling
    essentially unchanged and worsens several other common species.
- Decision:
  - Do not continue tuning environmental-neighbor k as the main research path.
    k=3 gives slightly better aggregate ranking; k=6 gives better block-79
    transfer; k=12 gives better calibration. These are small, coherent
    smoothing tradeoffs, not a breakthrough.
  - Keep k=6 as the most useful targeted coastal-transfer hybrid branch, and
    keep the spatial-only frozen-access model as the cleaner baseline branch.
  - The next meaningful framework step should address the source of the
    remaining ceiling: limited ecological/access support and missing relational
    structure, not another small cell-edge tweak.

Species relational structure next step:

- Rationale:
  - Aggregate AUPRC has barely moved across spatial residual, frozen access,
    and environmental-neighbor variants. This suggests the checklist covariates
    already capture most easy ranking signal, and the spatial-cell GNN is mostly
    redistributing residual corrections across blocks/species.
  - The next GNN-specific signal should come from the species side of the graph:
    species that co-occur under similar checklist/ecological contexts may share
    useful information, especially for lower-prevalence or geographically
    concentrated species.
  - This is closer to the original heterogeneous-graph goal than another
    spatial smoothing tweak: the model should learn from relationships among
    checklists, places, effort/access, and species, not only from cell adjacency.
- Initial implementation target:
  - Add an optional train-only species co-detection graph.
  - Build species-species edges from the training label matrix only, using
    normalized co-detection similarity.
  - Apply a small GCN over the learnable species embeddings before checklist x
    species scoring.
  - Keep the default off so all prior runs remain reproducible.
- First test should combine the cleaner spatial branch with species relational
  message passing:
  - frozen access
  - separated spatial channels
  - spatial-only cell graph first, not hybrid environmental edges
  - species co-detection graph with a small neighbor count
- Readout:
  - If species GCN improves macro AUPRC more than micro AUPRC, it is likely
    helping less-common species.
  - If it improves block/species failures without changing aggregate AUPRC much,
    it may still be useful for the framework.
  - If it mostly boosts common/generalist co-detection shortcuts and worsens
    block transfer, it should not be treated as a bias-correction component.
- Implementation notes:
  - `exp/ebird_spatial_gnn_baseline.py` now supports `--species-edge-mode
    codetection`, `--species-neighbors`, `--species-gcn-layers`, and
    `--species-gcn-dropout`.
  - Co-detection edges are built only from the training label matrix for the
    active split.
  - Saved run metadata records species graph mode, neighbor count, GCN layers,
    and species graph edge count so diagnostics can reconstruct the same graph.
  - Smoke test passed with a capped one-epoch run.

First species co-detection GCN command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_frozen_access_species_codetect_k10_l1 --gnn-mode residual --component-mode joint --spatial-channel-mode separated --species-edge-mode codetection --species-neighbors 10 --species-gcn-layers 1 --species-gcn-dropout 0.05 --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-access-bias-l2 0.001 --frozen-access-embeddings data/ebird/graph_top100_spatial_10x10/access_encoder/access_gcn_h64_l2_z64_cell_embeddings.npy
```

Then run:

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_frozen_access_species_codetect_k10_l1
```

```
python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_frozen_access_species_codetect_k10_l1
```

First species co-detection GCN result:

- `spatial_gcn_frozen_access_species_codetect_k10_l1`
  - micro AUROC 0.8918
  - micro AUPRC 0.5807
  - macro AUROC 0.8451
  - macro AUPRC 0.4155
  - ECE 0.0073
  - max bin error 0.0260
  - species calibration MAE 0.0142
- Compared with the spatial-only frozen-access branch:
  - micro AUPRC improves slightly from 0.5800 to 0.5807.
  - macro AUPRC improves slightly from 0.4151 to 0.4155.
  - calibration worsens: ECE 0.0056 to 0.0073 and species calibration MAE
    0.0133 to 0.0142.
- Effort-strata diagnostics:
  - High-effort gains remain and are slightly stronger in some strata:
    duration `121+` minutes (+0.0137), distance `(2,5]` km (+0.0134),
    distance `5+` km (+0.0125), traveling (+0.0105), and 2 observers
    (+0.0105).
  - Block 65 improves strongly (+0.0121 micro AUPRC).
  - Block 79 worsens sharply (-0.0090 micro AUPRC), worse than spatial-only,
    k=3 hybrid, and k=6 hybrid.
- Block/species diagnostics:
  - Block 65 improves substantially: mean block/species AUPRC +0.0163 with
    75 gains and 20 losses.
  - Block 31 remains near neutral: mean +0.0013.
  - Block 79 worsens: mean -0.0092 with 49 losses and 48 gains.
  - The species GCN helps some water/coastal species in block 79
    (Yellow-rumped Warbler, Great Black-backed Gull, Ring-billed Gull,
    Hooded Merganser, Double-crested Cormorant), but the largest block-79
    losses remain common/generalist or human-associated species: House Finch,
    House Sparrow, European Starling, Northern Mockingbird, Northern Cardinal,
    Mourning Dove, Carolina Wren, and Red-winged Blackbird.
- Interpretation:
  - The co-detection species graph adds real species-side signal, but the first
    k=10/layer-1 implementation appears to favor the large inland/mixed block
    65 more than the hard coastal block 79.
  - This is a plausible failure mode for train-only co-detection: the learned
    species graph can encode dominant checklist assemblages and common-species
    shortcuts rather than improving transfer to out-of-support coastal
    assemblages.
  - The result supports species relational structure as a useful comparison
    branch, but not yet as a bias-correction improvement.
- Decision:
  - Do not combine species co-detection GCN with hybrid environmental cell edges
    yet. That would mix two smoothing mechanisms before understanding either.
  - Run one conservative species-graph sensitivity test with fewer species
    neighbors. If smaller k reduces block-79 damage while preserving macro
    AUPRC, keep the species graph branch. If not, pause species co-detection
    and consider richer species metadata/traits or broader geography instead.

Conservative species co-detection sensitivity command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_frozen_access_species_codetect_k3_l1 --gnn-mode residual --component-mode joint --spatial-channel-mode separated --species-edge-mode codetection --species-neighbors 3 --species-gcn-layers 1 --species-gcn-dropout 0.05 --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-access-bias-l2 0.001 --frozen-access-embeddings data/ebird/graph_top100_spatial_10x10/access_encoder/access_gcn_h64_l2_z64_cell_embeddings.npy
```

Conservative species co-detection k=3 result:

- `spatial_gcn_frozen_access_species_codetect_k3_l1`
  - micro AUROC 0.8918
  - micro AUPRC 0.5809
  - macro AUROC 0.8450
  - macro AUPRC 0.4157
  - ECE 0.0067
  - max bin error 0.0254
  - species calibration MAE 0.0134
- Compared with k=10:
  - micro AUPRC improves slightly from 0.5807 to 0.5809.
  - macro AUPRC improves slightly from 0.4155 to 0.4157.
  - ECE improves from 0.0073 to 0.0067.
  - species calibration MAE improves from 0.0142 to 0.0134.
- Compared with spatial-only frozen access:
  - micro AUPRC improves from 0.5800 to 0.5809.
  - macro AUPRC improves from 0.4151 to 0.4157.
  - calibration is still worse than spatial-only frozen access.
- Interpretation:
  - Reducing species neighbors helps slightly. This is consistent with the
    concern that broader co-detection smoothing can over-emphasize dominant
    common assemblages.
  - The aggregate gain remains small, so the decision depends on effort-strata
    and block/species diagnostics. In particular, block 79 must not collapse the
    way it did under k=10.

Conservative species co-detection k=3 diagnostics:

- Effort-strata diagnostics:
  - High-effort gains remain and are slightly stronger than k=10 in some
    contexts: duration `121+` minutes (+0.0147 micro AUPRC), distance `(2,5]`
    km (+0.0141), traveling checklists (+0.0110), and 2 observers (+0.0111).
  - Block 65 remains strongly positive (+0.0119 micro AUPRC).
  - Block 31 remains mildly positive (+0.0042 micro AUPRC).
  - Block 79 worsens further: -0.0131 micro AUPRC and -0.0104 macro AUPRC.
    This is worse than k=10 co-detection, hybrid environmental-neighbor k=3/k=6,
    and spatial-only frozen access.
- Block/species diagnostics:
  - Block 65 remains positive: mean block/species AUPRC +0.0150 with 70 gains
    and 25 losses.
  - Block 31 remains near neutral to mildly positive: mean +0.0014 with
    58 gains and 37 losses.
  - Block 79 worsens materially: mean -0.0112, median -0.0047, with
    59 species losses and 38 gains.
  - Some coastal/water-associated block-79 species still benefit
    (Yellow-rumped Warbler, Ring-billed Gull, Great Black-backed Gull,
    Hooded Warbler, Red-breasted Nuthatch), but the common/generalist losses
    dominate the block-level result: House Finch, House Sparrow, European
    Starling, Mourning Dove, Northern Cardinal, Great Egret, Red-winged
    Blackbird, Carolina Wren, and Red-bellied Woodpecker.
- Decision:
  - Pause the simple co-detection species-GCN branch. It adds a small aggregate
    ranking gain, but it consistently worsens the hardest coastal block.
  - The failure mode is informative: train-only co-detection seems to encode
    dominant inland/common assemblage structure more than transferable
    ecological relationship structure.
  - Do not combine this branch with environmental cell edges yet. That would
    likely stack two smoothing mechanisms that both have block-specific
    tradeoffs.
  - If species relational structure is revisited, it should use a more
    ecologically constrained graph: taxonomy/traits, habitat guilds, migratory
    strategy, or species embeddings learned from a broader geographic dataset,
    rather than raw NC train co-detection alone.

Access-density auxiliary separated-channel command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_separated_channels_access_density_w0p01 --gnn-mode residual --component-mode joint --spatial-channel-mode separated --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-access-bias-l2 0.001 --access-density-loss-weight 0.01
```

Then repeat the same diagnostics:

```
python exp/diagnose_ebird_cell_residual_effort.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_separated_channels_access_density_w0p01 --min-cell-checklists 10
```

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_separated_channels_access_density_w0p01
```

```
python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_separated_channels_access_density_w0p01
```

First separated-channel command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name spatial_gcn_separated_channels_shared_access_l2_0p001 --gnn-mode residual --component-mode joint --spatial-channel-mode separated --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-access-bias-l2 0.001
```

Diagnostics after the separated-channel run:

```
python exp/diagnose_ebird_cell_residual_effort.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_separated_channels_shared_access_l2_0p001 --min-cell-checklists 10
```

Readout goal:

- Desired pattern: access-bias summaries correlate most strongly with effort and
  access predictors, while species-specific residual summaries correlate more
  with ecological/environmental gradients and observed species composition.
- Concerning pattern: species-specific residuals still dominate effort/access
  correlations, or access-bias terms mostly track environmental gradients. That
  would mean the architecture is still not separating bias and suitability in a
  useful way.

Next validation for the separated-channel branch:

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_separated_channels_shared_access_l2_0p001
```

```
python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_separated_channels_shared_access_l2_0p001
```

Diagnostics after a residual-regularized run:

```
python exp/diagnose_ebird_cell_residual_effort.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name RUN_NAME --min-cell-checklists 10
```

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name RUN_NAME
```

```
python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name RUN_NAME
```

After each run, compare to the 10x10 tabular MLP baseline using the matching
species metrics file.

Shared effort-bias comparison:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --baseline-dir data/ebird/baselines_10x10 --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/spatial_gnn_spatial_gcn_joint_scaled_effort_shared_l2_0p001_test_species_metrics.csv --output data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/joint_scaled_effort_shared_l2_0p001_graph_vs_tabular_species_fair_10x10.csv
```

Low-rank effort-bias comparison:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --baseline-dir data/ebird/baselines_10x10 --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/spatial_gnn_spatial_gcn_joint_scaled_effort_lowrank8_l2_0p001_test_species_metrics.csv --output data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/joint_scaled_effort_lowrank8_l2_0p001_graph_vs_tabular_species_fair_10x10.csv
```

Then run effort-strata and block-by-species diagnostics.

Shared effort-bias diagnostics:

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_joint_scaled_effort_shared_l2_0p001
```

```
python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_joint_scaled_effort_shared_l2_0p001
```

Low-rank effort-bias diagnostics:

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_joint_scaled_effort_lowrank8_l2_0p001
```

```
python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_joint_scaled_effort_lowrank8_l2_0p001
```

Residual maps for whichever effort-bias run looks most promising:

```
python exp/plot_ebird_spatial_gnn_residual_maps.py --graph-dir data/ebird/graph_top100_spatial_10x10 --run-name RUN_NAME --species "Red-headed Woodpecker" "Eastern Meadowlark" "House Sparrow" "Bufflehead" "Pied-billed Grebe" "Swamp Sparrow" "Wood Duck" "Double-crested Cormorant" "Killdeer" "Black-and-white Warbler" --boundary data/boundaries/nc_state_boundary.gpkg
```

Step 4 placeholder, richer graph structure:

- If the residual GNN gain is stable, add graph structure incrementally and
  compare against the same diagnostics:
  - species co-detection edges or species-similarity edges
  - environmental-neighbor cell edges, not only queen spatial adjacency
  - protocol/effort context nodes or edge attributes
  - locality nodes with train-only/leave-one-out safeguards
  - observer nodes only later, with privacy, leakage, and transfer constraints
- Each new relation type should be evaluated against the same all-pairs target,
  species-level deltas, residual maps, and effort-strata calibration checks.

Residual/gated grid search:

- `exp/run_ebird_spatial_gnn_grid.py` wraps
  `exp/ebird_spatial_gnn_baseline.py` and runs named residual/gated
  configurations. It skips existing summary JSONs by default, so it can be
  stopped and resumed.
- The first grid should stay small and targeted:

  - `--gnn-mode residual` and `--gnn-mode gated`
  - cell hidden dimensions 32 and 64
  - one and two spatial-cell GCN layers
  - weight decay 1e-4 and 1e-3
  - gated init biases -2 and -3

- Interpretation targets:

  - preserve or improve micro AUPRC relative to the stronger hybrid (0.5895)
    and RBF spatial residual (0.5910)
  - preserve or improve macro AUPRC relative to the stronger hybrid (0.4242)
    and RBF spatial residual (0.4248)
  - keep ECE and species calibration MAE close to the stronger hybrid/RBF
    residual, because the bias/effort goal needs sane detection probabilities
  - reduce the large species-level losses for Red-headed Woodpecker, Swamp
    Sparrow, Wood Duck, Green Heron, Ovenbird, and Wood Thrush

Grid dry-run command:

```
python exp/run_ebird_spatial_gnn_grid.py --dry-run
```

Conservative first grid command:

```
python exp/run_ebird_spatial_gnn_grid.py --max-runs 8
```

Full default grid command:

```
python exp/run_ebird_spatial_gnn_grid.py
```

Visual diagnostics:

- `exp/plot_ebird_spatial_gnn_grid.py` reads the spatial GNN output directory,
  all-species link baseline summaries, the tabular MLP summary, calibration
  tables, and any graph-vs-tabular species comparison CSVs.
- It writes:

  - `spatial_gnn_run_summary.csv`
  - `run_metric_bars.png`
  - `ranking_calibration_tradeoff.png`
  - `spatial_gnn_calibration_curves.png`
  - one top/bottom species AUPRC-delta plot for each available comparison CSV

Visual diagnostic command:

```
python exp/plot_ebird_spatial_gnn_grid.py
```

After each promising grid run, compare species-level deltas:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/spatial_gnn_baselines/spatial_gnn_<RUN_NAME>_test_species_metrics.csv
```

Concrete next comparison commands:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/spatial_gnn_baselines/spatial_gnn_spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001_test_species_metrics.csv
```

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/spatial_gnn_baselines/spatial_gnn_spatial_gcn_gated_h128_l2_z128_cell64_cl1_wd0p0001_gbm2_test_species_metrics.csv
```

Preferred short-output comparison commands:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/spatial_gnn_baselines/spatial_gnn_spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001_test_species_metrics.csv --output data/ebird/graph_top100_spatial/spatial_gnn_baselines/residual_primary_graph_vs_tabular_species.csv
```

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/spatial_gnn_baselines/spatial_gnn_spatial_gcn_gated_h128_l2_z128_cell64_cl1_wd0p0001_gbm2_test_species_metrics.csv --output data/ebird/graph_top100_spatial/spatial_gnn_baselines/gated_gbm2_graph_vs_tabular_species.csv
```

Species diagnostic command:

```
python exp/diagnose_ebird_spatial_gnn_species.py --graph-dir data/ebird/graph_top100_spatial --comparison-csv data/ebird/graph_top100_spatial/spatial_gnn_baselines/residual_primary_graph_vs_tabular_species.csv --comparison-csv data/ebird/graph_top100_spatial/spatial_gnn_baselines/gated_gbm2_graph_vs_tabular_species.csv --boundary data/boundaries/nc_state_boundary.gpkg
```

Species diagnostic outputs:

- `focus_species_model_deltas.csv`: residual and gated species-level deltas
  against the tabular MLP.
- `focus_species_split_summary.csv`: train/test positives, prevalence, and
  spatial coverage by focus species.
- `focus_species_test_strata.csv`: held-out prevalence by protocol, duration,
  distance, observer count, and spatial block.
- `focus_species_test_covariates.csv`: held-out positive-vs-background
  covariate means.
- `focus_species_auprc_delta.png` and `focus_species_test_coverage.png`.
- one held-out positive-location map per focus species.

Residual probability-difference map command:

```
python exp/plot_ebird_spatial_gnn_residual_maps.py --graph-dir data/ebird/graph_top100_spatial --run-name spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001 --species "Black-and-white Warbler" "Eastern Meadowlark" "Red-headed Woodpecker" "Swamp Sparrow" "Wood Duck" "Green Heron" --boundary data/boundaries/nc_state_boundary.gpkg
```

Residual map outputs:

- one `*_residual_probability_delta.png` map per selected species under
  `data/ebird/graph_top100_spatial/spatial_gnn_baselines/diagnostics/residual_maps`
- `spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001_residual_probability_summary.csv`
  with mean base probability, full probability, residual delta, and
  positive-vs-negative delta summaries.

Effort-strata diagnostic command:

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial --spatial-run-name spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001
```

Effort-strata diagnostic outputs:

- `effort_strata_metrics.csv`: tabular MLP and spatial GNN metrics within each
  protocol, duration, distance, observer-count, and spatial-block stratum.
- one `*_strata_deltas.png` plot per stratum type under
  `data/ebird/graph_top100_spatial/spatial_gnn_baselines/diagnostics/effort_strata`
- `effort_strata_metadata.json` with the spatial run and retrained tabular MLP
  settings.

Then rerun:

```
python exp/plot_ebird_spatial_gnn_grid.py
```

Species-level comparison for the best residual grid run:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/spatial_gnn_baselines/spatial_gnn_spatial_gcn_residual_h128_l2_z128_cell64_cl1_wd0p0001_test_species_metrics.csv
```

The `wd0p001` run has slightly higher micro AUPRC, but `wd0p0001` has slightly
better macro AUPRC, ECE, and species calibration MAE. Use `wd0p0001` as the
first species-level diagnostic unless later plots show a specific reason to
prefer `wd0p001`.

Recommended residual spatial GNN command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial --run-name spatial_gcn_residual_h128_l2_z128_cell64_l2 --gnn-mode residual --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 2 --dropout 0.10 --spatial-grid-size-m 25000
```

Recommended gated spatial GNN command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial --run-name spatial_gcn_gated_h128_l2_z128_cell64_l2 --gnn-mode gated --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 2 --dropout 0.10 --spatial-grid-size-m 25000 --gate-init-bias -2
```

Residual comparison command:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/spatial_gnn_baselines/spatial_gnn_spatial_gcn_residual_h128_l2_z128_cell64_l2_test_species_metrics.csv
```

Gated comparison command:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/spatial_gnn_baselines/spatial_gnn_spatial_gcn_gated_h128_l2_z128_cell64_l2_test_species_metrics.csv
```

Spatial-neighbor bridge implementation:

- `exp/ebird_graph_all_species_baseline.py` supports
  `--feature-augmentation spatial-neighbor-scalars` and
  `--feature-augmentation spatial-neighbor`.
- `spatial-neighbor-scalars` adds four scalar checklist features:

  - `spatial_neighbor_train_checklists_log1p`
  - `spatial_neighbor_train_cells_log1p`
  - `spatial_neighbor_species_rate_mean`
  - `spatial_neighbor_mean_distance_ratio`

- `spatial-neighbor` additionally adds smoothed per-species spatial-neighbor
  prior logits, with the prior-logit weight initialized at zero by default.
- Training rows use leave-one-out adjustment for the checklist's own grid cell.
  Held-out rows use only training split grid-cell aggregates.
- Default settings use 25 km training grid cells, a 75 km neighbor radius, a
  50 km exponential distance-decay scale, at least 3 nearby training cells, and
  prior smoothing of 20 checklist-equivalents.

Recommended spatial-neighbor scalar diagnostic:

```
python exp/ebird_graph_all_species_baseline.py --graph-dir data/ebird/graph_top100_spatial --architecture hybrid --feature-augmentation spatial-neighbor-scalars --run-name hybrid_h128_l2_z128_spatial_neighbor_scalars --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --dropout 0.10 --spatial-grid-size-m 25000 --spatial-neighbor-radius-m 75000 --spatial-neighbor-decay-m 50000 --spatial-neighbor-min-cells 3 --prior-smoothing 20
```

Spatial-neighbor scalar result:

- Micro AUROC 0.8859, micro AUPRC 0.5692.
- Macro AUROC 0.8406, macro AUPRC 0.4119.
- ECE 0.0057, max probability-bin error 0.0424.
- Species calibration MAE 0.0234.
- This is better than the locality/grid aggregate variants but worse than the
  plain stronger hybrid on ranking and species calibration.

Spatial-neighbor prior-logit diagnostic:

```
python exp/ebird_graph_all_species_baseline.py --graph-dir data/ebird/graph_top100_spatial --architecture hybrid --feature-augmentation spatial-neighbor --run-name hybrid_h128_l2_z128_spatial_neighbor_w0 --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --dropout 0.10 --spatial-grid-size-m 25000 --spatial-neighbor-radius-m 75000 --spatial-neighbor-decay-m 50000 --spatial-neighbor-min-cells 3 --prior-smoothing 20 --prior-logit-weight 0
```

Spatial-neighbor prior-logit result:

- Micro AUROC 0.8859, micro AUPRC 0.5710.
- Macro AUROC 0.8402, macro AUPRC 0.4136.
- ECE 0.0063, max probability-bin error 0.0478.
- Species calibration MAE 0.0231.
- Learned prior-logit weight 0.2105.
- This is the best spatial-neighbor variant so far, but it still trails the
  plain stronger hybrid on aggregate ranking and species calibration.

Comparison command:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/all_species_link_baselines/all_species_link_hybrid_h128_l2_z128_spatial_neighbor_scalars_test_species_metrics.csv
```

Explicit spatial residual bridge:

- `exp/ebird_graph_all_species_baseline.py` supports
  `--spatial-residual rbf`.
- This keeps the stronger hybrid bridge as the main ecology/effort/species
  interaction model and adds a separate additive species-specific spatial
  residual:

  \[
  \operatorname{logit} P(y_{c,j}=1)
  =
  \text{hybrid}_{j}(x_c)
  +
  r_j(s_c)
  \]

- The residual \(r_j(s_c)\) is a linear combination of fixed radial-basis
  spatial features. The RBF centers are laid out over the training-checklist
  spatial extent, features are standardized with training checklists only, and
  the residual head is initialized at zero. This makes it an explicit correction
  to the bridge rather than a replacement for effort/ecology covariates.
- This benchmark answers: how much smooth spatial structure remains after the
  current effort/ecology/species bridge? If this improves clustered species but
  hurts spatial transfer or calibration, the eventual GNN must be constrained so
  it does not just learn observer geography.

Recommended spatial residual bridge command:

```
python exp/ebird_graph_all_species_baseline.py --graph-dir data/ebird/graph_top100_spatial --architecture hybrid --run-name hybrid_h128_l2_z128_spatial_residual_rbf --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --dropout 0.10 --spatial-residual rbf --spatial-residual-grid-per-dim 12 --spatial-residual-length-scale-m 100000
```

Comparison command:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial/all_species_link_baselines/all_species_link_hybrid_h128_l2_z128_spatial_residual_rbf_test_species_metrics.csv
```

Spatial residual result:

- Micro AUROC 0.8933, micro AUPRC 0.5910.
- Macro AUROC 0.8458, macro AUPRC 0.4248.
- ECE 0.0108, max probability-bin error 0.0382.
- Species calibration MAE 0.0144.
- This was the best aggregate ranking model before the tuned spatial-cell GNN
  grid, but its calibration is worse than the plain stronger hybrid. Treat it
  as the final non-GNN benchmark for the first true GNN.

## Current Framework Checkpoint

The repeated result across the frozen-access spatial GNN, environmental-neighbor
edges, separated spatial channels, access-density supervision, and raw
co-detection species GCNs is that aggregate metrics move only modestly while
the same held-out coastal block remains fragile. This suggests the immediate
problem is not a lack of small architecture variants. It is likely a support and
regime-transfer problem: the held-out coastal/island block has species,
water/coast, elevation/canopy, and access structure that may not have close
training analogs.

Current interpretation:

- Keep `spatial_gcn_frozen_access_h64_l2_z64` as the clean spatial-cell GNN
  baseline.
- Keep `spatial_gcn_frozen_access_env_edges_k6` as a targeted comparison for
  environmentally similar cell transfer.
- Pause the raw co-detection species GCN branch. It adds a small aggregate
  ranking gain, but worsens block 79 and appears to transfer common checklist
  assemblage structure more than robust ecological structure.
- Do not tune the split away from block 79 yet. It is useful as a stress test,
  but we need to diagnose whether it is out of support before using it to choose
  architectures.

Next diagnostic step:

- `exp/diagnose_ebird_regime_support.py` summarizes held-out block/cell support
  in coastal, waterbody, elevation, canopy, and effort/access space.
- The goal is to determine whether block 79 is genuinely out of support, whether
  environmental-neighbor edges are connecting it to useful training analogs, and
  whether future work should prioritize:
  - richer coastal/water/habitat covariates,
  - a broader geographic training domain with more coastal analogs,
  - a revised blocked validation design with multiple coastal and inland test
    blocks, or
  - ecologically constrained species graphs rather than raw co-detection.

Regime-support diagnostic command:

```
python exp/diagnose_ebird_regime_support.py --graph-dir data/ebird/graph_top100_spatial_10x10
```

Primary outputs:

- `data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/diagnostics/regime_support/regime_support_test_block_summary.csv`
- `data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/diagnostics/regime_support/regime_support_test_cells.csv`
- `data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/diagnostics/regime_support/regime_support_top_species_by_test_block.csv`

How to read this diagnostic:

- High `nearest_train_ecology_distance` means held-out cells differ from the
  training cells in canopy, elevation, waterbody distance, and coastline
  distance after standardization.
- High `nearest_train_access_distance` means the held-out cells differ in
  protocol/effort structure, not just habitat.
- Large `nearest_train_spatial_distance_m` is expected for held-out blocks, but
  it matters most when paired with high ecological/access distance.
- If block 79 is high on ecological distance, the next framework step should be
  richer coastal/water/habitat features or broader geography.
- If block 79 is ecologically similar but access-different, the next step should
  be a stronger explicit access/bias component.
- If block 79 is neither ecologically nor access out-of-support, then the model
  architecture or loss/objective is more likely the bottleneck.

Diagnostic revision after first run:

- The first regime-support run showed block 79 as fully coastal and strongly
  near-water, with top positives dominated by coastal/water-associated species
  such as Double-crested Cormorant, Red-winged Blackbird, Boat-tailed Grackle,
  Laughing Gull, and American Herring Gull.
- However, the first script version also reported non-test blocks in the
  held-out summary because 25 km regime cells can straddle validation block
  boundaries and were labeled by the modal block across all checklists.
- `exp/diagnose_ebird_regime_support.py` was revised so training support cells
  are summarized from train rows only, and held-out cells are summarized by
  `(spatial_block, spatial_cell)` from test rows only.
- Rerun the command above before interpreting the ecology/access support
  distances. The top-species signal from the first run is still useful, but the
  block-level support distances should come from the revised script.

Corrected regime-support result:

- Held-out block 79 is the only coastal test block:
  - coastal rate 1.0000
  - near-water rate 0.7611
  - mean coastline distance about 1.03 km
  - mean waterbody distance about 1.36 km
- Block 79 has the highest access/effort regime distance from training cells:
  - nearest train access distance 0.8826
  - block 31 is 0.4329
  - block 65 is 0.3561
- Block 79 is not the most ecologically distant block by the current broad
  raster covariates:
  - nearest train ecology distance 0.4908
  - block 31 is higher at 0.6112
  - block 65 is lower at 0.3521
- Block 79 top positives are strongly coastal/water-associated: Double-crested
  Cormorant, Red-winged Blackbird, Boat-tailed Grackle, Laughing Gull, and
  American Herring Gull.

Interpretation:

- The repeated block-79 failure is probably not just an ecological covariate
  out-of-support problem. It is more likely a combination of:
  - coastal/water species assemblage,
  - distinct access/effort geography,
  - and a validation design where the only fully coastal block is held out.
- The existing continuous water/coast distance covariates are present, but the
  model may still need explicit coastal/access regime structure so it does not
  treat coastal observer geography as an unconstrained spatial residual.
- The fact that block 79 species have substantial train detections means this is
  not simply a no-training-data species problem. It is a checklist-regime and
  spatial-transfer problem.

Next step:

- Add a targeted coastal/access-regime diagnostic or model feature set before
  more GNN architecture changes.
- Candidate derived features:
  - coastal indicator, for example distance to coastline <= 25 km
  - near-water indicator, for example distance to waterbody <= 2.5 km
  - log coastline distance and log waterbody distance, if not already modeled
    in transformed form
  - coastal-by-effort interactions, especially coastal x traveling, coastal x
    effort distance, coastal x duration, and near-water x protocol
- Then compare the tabular MLP and clean frozen-access spatial GNN with and
  without these derived regime features. If the tabular model improves block 79,
  the missing piece is feature representation rather than message passing. If
  the tabular model does not improve but the GNN does, the graph structure is
  adding useful transfer beyond explicit covariates.

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
- strata based on environmental covariates and common-species prevalence where
  feasible, so the held-out region is not an ecologically unusual leftover
- temporal holdouts by year or season
- species-stratified metrics so common species do not dominate
- rare-species metrics for species with enough detections

Key metrics:

- held-out checklist species prediction AUROC/AUPRC
- calibration by predicted-probability bin
- calibration by effort stratum, including protocol, duration, distance, and
  number of observers
- performance by county or spatial block
- Wood Thrush-specific comparison against existing IPPP/NIPPP outputs
- sensitivity to protocol and effort filters

Immediate validation next steps:

1. Add spatial-stratified blocked validation to the tabular baseline. The split
   should hold out whole geographic blocks while greedily matching the full
   dataset on checklist count, effort variables, raster covariates, and
   top-species prevalence.
2. Compare temporal and spatial-stratified results for effort, ecology, and
   combined feature sets. A large drop under spatial validation would indicate
   that the temporal split is benefiting from repeated geography or observer
   structure.
3. Add calibration outputs by predicted probability, protocol, duration, and
   checklist effort strata. The tabular baseline now writes
   `*_calibration.csv` files with mean predicted probability, observed
   detection rate, and calibration error for each bin/stratum; the summary JSON
   includes expected calibration error over predicted-probability bins.
4. Add a small nonlinear MLP tabular baseline before graph construction. This
   tests whether the remaining ranking/calibration gaps are due to linear
   underfitting before adding graph structure.

Spatial-stratified baseline commands:

```
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --feature-set effort --split spatial-stratified --spatial-blocks-per-dim 8 --test-fraction 0.2 --epochs 50
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --feature-set ecology --split spatial-stratified --spatial-blocks-per-dim 8 --test-fraction 0.2 --epochs 50
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --feature-set both --split spatial-stratified --spatial-blocks-per-dim 8 --test-fraction 0.2 --epochs 50
python exp/compare_ebird_tabular_baselines.py --top-species 100 --split spatial-stratified
```

Nonlinear MLP (Multi-Layer Perceptron) tabular baseline commands:

```
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --feature-set effort --model mlp --split spatial-stratified --spatial-blocks-per-dim 8 --test-fraction 0.2 --epochs 50 --hidden-dim 64 --hidden-layers 1 --dropout 0.10
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --feature-set ecology --model mlp --split spatial-stratified --spatial-blocks-per-dim 8 --test-fraction 0.2 --epochs 50 --hidden-dim 64 --hidden-layers 1 --dropout 0.10
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --feature-set both --model mlp --split spatial-stratified --spatial-blocks-per-dim 8 --test-fraction 0.2 --epochs 50 --hidden-dim 64 --hidden-layers 1 --dropout 0.10
python exp/compare_ebird_tabular_baselines.py --top-species 100 --split spatial-stratified --model mlp
python exp/compare_ebird_calibration.py --top-species 100 --split spatial-stratified --model mlp
```

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
6. Add spatial-stratified blocked validation to the tabular baseline.
7. Add calibration summaries by probability and effort strata.
8. Add small nonlinear MLP tabular baseline.
9. Build graph-ready checklist/species node tables and positive/negative edge
   tables.
10. Train a non-message-passing embedding/link baseline on the graph dataset.
11. Train an all-species-per-checklist graph bridge objective.
12. Add graph construction and a simple heterogeneous GNN.
13. Compare Wood Thrush predictions to existing single-species IPPP/NIPPP runs.
