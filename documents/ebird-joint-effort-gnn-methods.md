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
   improves calibration without changing AUROC/AUPRC ranking.
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
  species-embedding interaction term. It is the recommended next bridge
  architecture to test.

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
- Hybrid species-level gains over tabular are now real for some species. The
  largest AUPRC gains include Bald Eagle, Hooded Warbler, Double-crested
  Cormorant, Black-and-white Warbler, Brown Thrasher, Red-shouldered Hawk,
  American Robin, Ruby-throated Hummingbird, House Finch, Eastern Towhee,
  Pileated Woodpecker, and Yellow-throated Warbler.
- The largest remaining hybrid losses include Eastern Meadowlark, Northern
  Rough-winged Swallow, Mallard, Red-headed Woodpecker, Hooded Merganser, Wood
  Duck, Pied-billed Grebe, American Herring Gull, Blue Grosbeak, Swamp Sparrow,
  Great Egret, and Ring-billed Gull.
- Species calibration is also now comparable across the tabular and bridge
  outputs. The tabular MLP species calibration MAE is 0.0132, slightly better
  than the hybrid bridge at 0.0144.
- Because the hybrid bridge nearly matches the tabular MLP using only checklist
  covariates and species interactions, a full GNN should focus on adding
  information not present in the current feature matrix: locality/hotspot
  repeated-visit structure, spatial neighbor edges, observer effects, or
  checklist co-detection context.

Near-term next steps:

1. Add species-level calibration to the tabular MLP metrics so tabular and graph
   bridge outputs can be compared on the same species calibration diagnostics.
   The tabular metrics should include per-species `mean_predicted` and
   `calibration_error`, with `species_calibration_mae` in the summary JSON.
2. Run one stronger hybrid bridge as a capacity check:

```
python exp/ebird_graph_all_species_baseline.py --graph-dir data/ebird/graph_top100_spatial --architecture hybrid --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --dropout 0.10
```

3. If the stronger hybrid matches or beats the tabular MLP, the remaining issue
   was bridge capacity rather than graph structure. If it still trails, build a
   locality/spatial-neighbor enriched bridge before adding message passing.
4. Prioritize locality/spatial structure before observer structure. Observer
   effects are likely strong but can dominate ecology and should be introduced
   carefully after cleaner spatial/locality signals are tested.
5. Candidate relational features for the enriched bridge:

   - train-only locality or hotspot visit count
   - train-only locality species detection rates
   - spatial-neighborhood checklist density
   - spatial-neighborhood species detection rates
   - repeated-visit count by locality/month or locality/season
   - local species richness or co-detection summaries fit from training data

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
