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
| Stronger hybrid all-species bridge | 0.8935 | 0.5895 | 0.8476 | 0.4242 | 0.0038 | 0.0136 |
| Stronger hybrid + locality/spatial prior | 0.8605 | 0.5143 | 0.8059 | 0.3455 | 0.0130 | 0.0207 |
| Stronger hybrid + locality/spatial scalars | 0.8671 | 0.5325 | 0.8217 | 0.3790 | 0.0181 | 0.0315 |
| Stronger hybrid + locality/spatial prior, weight init 0 | 0.8694 | 0.5313 | 0.8209 | 0.3725 | 0.0195 | 0.0252 |
| Stronger hybrid + spatial-neighbor scalars | 0.8859 | 0.5692 | 0.8406 | 0.4119 | 0.0057 | 0.0234 |
| Stronger hybrid + spatial-neighbor prior, weight init 0 | 0.8859 | 0.5710 | 0.8402 | 0.4136 | 0.0063 | 0.0231 |
| Stronger hybrid + RBF spatial residual | 0.8933 | 0.5910 | 0.8458 | 0.4248 | 0.0108 | 0.0144 |

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
- The explicit RBF spatial residual is the best aggregate ranking model so far:
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

Near-term next steps:

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
4. Next: build a locality/spatial-neighbor enriched bridge before adding message
   passing. This should test whether relational information helps when added as
   train-only aggregate features.
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

Recommended next modeling step:

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
5. Next: build the first true message-passing GNN and compare it against both
   the plain stronger hybrid and the RBF spatial residual. The GNN needs to
   improve ranking without worsening calibration enough to undermine the
   bias/effort modeling goal.

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
- This is the best aggregate ranking model so far, but its calibration is worse
  than the plain stronger hybrid. Treat it as the final non-GNN benchmark for
  the first true GNN.

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
