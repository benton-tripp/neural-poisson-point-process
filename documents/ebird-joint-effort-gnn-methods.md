# eBird Joint Effort and Multi-Species SDM Plan

## Goal

Develop a generally applicable U.S. eBird workflow for joint species
distribution modeling that explicitly represents observer effort and imperfect
detection. North Carolina eBird 2020-2023 is the initial development and
validation testbed, not the geographic scope of the intended method. The first
target is a reusable intermediate dataset, not a final model:

- checklist/location-time nodes with effort, temporal, spatial, and environmental features
- species nodes with taxonomic identifiers and summary frequencies
- detection edges connecting species to checklists or aggregated location-time cells

This representation can support both a joint neural point-process model and a
heterogeneous graph neural network for complete-checklist detection modeling.
Although the broader SDM motivation starts from presence-only citizen-science
data, the current eBird workflow uses complete checklists, so unreported modeled
species on retained checklists are treated as observed non-detections under an
explicit observation/effort process rather than as arbitrary pseudo-absences.

## Current Analysis Checkpoint

The locality-season latent repeated-visit model remains the correct primary
scientific path. The NC top-100 analysis is a development and validation
testbed for a transferable framework, not an attempt to optimize only these
species or this geography. The current decision is:

- Keep the unregularized two-component bridge
  (`two_component_checklist_detection_e10_d20`) as the strongest fair
  checklist-level predictive benchmark. It is not a latent occupancy model.
- Keep the repeated-visit likelihood as the primary structural model because
  it explicitly separates locality-season availability (`psi`) from
  checklist-level conditional detection (`p`) and uses complete-checklist
  temporal replication rather than arbitrary pseudo-absences.
- Use `latent_marginal_all_pairs` as the fair held-out checklist metric.
  Label-informed posterior and known-positive-group conditional metrics remain
  diagnostics only.
- Retain
  `latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025`
  as the current parsimonious latent sensitivity. Its fair prior-marginal
  metrics are AUPRC 0.56862, BCE 0.29629, calibration error 0.01488, ECE
  0.01834, and focus-season weighted absolute error 0.01596.
- The completed `both` species-season run is effectively tied on pooled
  ranking but is not a clear replacement. Relative to the detection-only
  species-season run, it changes AUPRC by only +0.00002, improves BCE by
  0.00035, calibration error by 0.00041, ECE by 0.00080, and max-bin error by
  0.00495, but slightly worsens focus-season weighted error from 0.01596 to
  0.01604 and availability positive-triplet AUPRC from 0.71673 to 0.71589.
- Giving both components species-season offsets redistributes seasonal signal
  without fixing the persistent species failures. Its availability-season
  parameter RMS is 0.41067 and detection-season RMS is 0.52154, compared with
  detection-season RMS 0.55843 and no availability-season offset in the
  detection-only run. This is evidence of added component redundancy, not a
  new biological result.
- Persistent losses versus the bridge remain concentrated in Tree Swallow,
  Hooded Merganser, Red-breasted Nuthatch, Mallard, Bald Eagle,
  White-throated Sparrow, Bufflehead, Swamp Sparrow, Double-crested Cormorant,
  Downy Woodpecker, and related species. The pattern spans multiple ecological
  groups and is not solved by moving seasonal flexibility between `psi` and
  `p`.
- Close the current species-season placement/L2 axis. Do not add another
  offset strength, GNN layer, or post-hoc calibrator in response to this run.
- The next question is component identifiability and replication support:
  determine whether failures arise from the availability component,
  conditional detection, or locality-seasons with weak variation in visits,
  dates, protocols, observers, and effort. The latent training script now
  writes compact component-by-season and component-by-support diagnostics for
  this purpose.
- The first component-support run shows that fair group-level any-detection
  error generally improves with stronger temporal and effort diversity, while
  checklist-level marginal detection becomes underpredicted in the most
  intensively sampled groups. Conditional `p` is approximately aligned inside
  high-support known-positive groups, although that diagnostic is
  label-informed. This points more toward availability/site heterogeneity and
  observer-geography concentration than a simple failure of the checklist
  effort coefficients.
- The completed strict-support sensitivity required at least five distinct
  dates and three duration bins, retaining 11,243 training groups, 4,564 test
  groups, and 99,471 test checklists. Group any-detection calibration is nearly
  exact in the strongest strata (`10+` dates or `11+` checklists), but
  medium-support groups still overpredict any-detection while high-support
  checklist frequency remains underpredicted. Filtering improves some
  component behavior but is not a complete solution and should not replace the
  broader target population.
- The pairwise co-detection diagnostic confirmed the likelihood-assumption
  problem. In every replication-support stratum, observed repeated
  co-detection exceeded the model-implied conditionally independent prediction.
  The overall strict-support weighted pairwise signed error was -0.03335; the
  `11+` checklist stratum was -0.03360, `10+` dates was -0.03370, one-observer
  groups were -0.03464, and 3-5 observer groups were -0.04230. This is direct
  evidence for excess repeated-detection dependence.
- The next model change is therefore a transferable detection-overdispersion
  term, not stricter filtering, another species-season offset, a GNN layer, or
  post-hoc calibration. The latent script now supports an optional
  logistic-normal shared detection frailty within each locality-season/species
  group via `--detection-frailty-mode global|species`.
- The completed global-frailty test validates that direction. Relative to the
  same strict-support independent-detection run:
  - overall weighted pairwise co-detection error improved from -0.03335 to
    -0.01562
  - 90 of 100 species had smaller absolute species-level pairwise error
  - fair checklist calibration error improved from 0.01361 to 0.00888, ECE
    from 0.01741 to 0.01337, and max-bin error from 0.06245 to 0.04768
  - fair checklist AUPRC changed only modestly from 0.57498 to 0.57279, while
    BCE improved slightly from 0.30566 to 0.30519
  - focus-species season weighted absolute error improved from 0.01498 to
    0.01311
  - overall fair group any-detection probability was nearly exact: observed
    0.41331 versus predicted 0.41551
  - the learned global logistic-normal frailty scale was 1.07439 logit units
- Mean latent availability rose from 0.50488 to 0.53329 while the observed
  any-detection rate was 0.41331. This is not direct occupancy
  miscalibration: an observed-positive locality-season is a lower bound on
  availability, and frailty explicitly permits available groups to have no
  detections. The fair check is the combined any-detection probability, which
  remained calibrated. Availability must instead be judged through transfer,
  phenology, environmental response, and sensitivity stability.
- The completed independently regularized species-frailty run did not improve
  on global frailty. Relative to the global run, overall weighted pairwise
  error was effectively unchanged (-0.01566 versus -0.01562), fair checklist
  AUPRC changed from 0.57279 to 0.57249, overall any-detection error increased
  from 0.00220 to 0.00325, and focus-season weighted error increased from
  0.01311 to 0.01355. Absolute species-level pairwise error improved for only
  37 of 100 species, and the median species became slightly worse.
- The learned species scales remained tightly clustered: mean 1.03631, RMS
  1.04000, implied standard deviation 0.08755, and maximum 1.22293. The
  existing `species` parameterization shrinks each absolute scale toward zero;
  it does not preserve the supported global frailty while partially pooling
  species departures. Close that formulation as a non-promoted sensitivity.
- The completed hierarchical-frailty run also did not beat global frailty.
  Relative to global, fair checklist AUPRC declined from 0.57279 to 0.57195,
  BCE increased from 0.30519 to 0.30538, overall any-detection error increased
  from 0.00220 to 0.00372, and focus-season weighted error increased from
  0.01311 to 0.01415. Overall pairwise error improved only from -0.01562 to
  -0.01535.
- Hierarchical scales were bounded but materially more variable (mean 1.02404,
  standard deviation 0.17436, range 0.55795-1.44796). Absolute species-level
  pairwise error improved for only 39 of 100 species and the median species
  worsened. The additional heterogeneity therefore did not yield a broad
  held-out benefit.
- Apply the stopping rule: promote the simpler global-frailty likelihood as
  the current structural latent model and close the frailty variance axis.
  Independently regularized and hierarchical species frailty remain documented
  non-promoted sensitivities.
- The promoted global model was rerun as
  `latent_strict_frailty_global_availdiag` to write compact held-out
  group-level availability artifacts. It reproduced the promoted checkpoint
  exactly: fair checklist AUPRC 0.57279, BCE 0.30519, calibration error
  0.00888, ECE 0.01337, observed/predicted group any-detection 0.41331/0.41551,
  weighted pairwise error -0.01562, and global frailty scale 1.07439.
- The first held-out availability audit is a qualified pass, not a final
  validation. Wood Thrush has near-zero winter behavior (observed any
  detection 0, `psi` 0.04767, predicted any detection 0.00765) and a clear
  early-breeding peak. Green Heron likewise falls to `psi` 0.03989 in winter
  and rises to 0.38367/0.42488 in the breeding windows. Northern Cardinal
  remains consistently high across seasons with fair observable errors within
  about +/-0.009. These are the expected broad phenology contrasts.
- Coarse five-bin environmental responses also retain useful shape. Across the
  ten focus species, mean observable-response Spearman correlations are about
  0.856-0.910 for elevation, coastline distance, waterbody distance, and
  canopy; mean species-level weighted observable MAE is about 0.022-0.027.
  This is encouraging but is only an internal binned-response check, not
  external ecological validation.
- Important failures remain. Great Egret late-breeding and fall-migration
  observable any-detection are overpredicted by about 0.089. Black-and-white
  Warbler has winter `psi` 0.25682 despite an observed-positive rate of
  0.08061, while its combined predicted any-detection error is only +0.02616;
  this is a direct example of availability/detection tradeoff hidden by an
  acceptable observable prediction. Wood Thrush observable any-detection is
  underpredicted by about 0.029-0.044 outside winter.
- High-support averages are mostly bounded, but individual zero-detection
  groups expose severe localized errors. Some Northern Cardinal, Eastern
  Towhee, Great Egret, and Double-crested Cormorant groups with many visits
  receive prior any-detection probabilities above 0.9. Across the ten focus
  species, 130 high-support zero-detection groups exceed 0.8 and 21 exceed
  0.9; the latter are concentrated in Northern Cardinal (9), Eastern Towhee
  (7), Great Egret (3), and Double-crested Cormorant (2). One audited Northern
  Cardinal case had 61 test visits on 54 dates, zero detections, and predicted
  any detection near 0.996; the same locality and observer had historical
  same-season detections, but the immediately preceding year also had zero.
  This points to temporal/locality and observer-history structure that the
  current ecology-plus-effort predictors do not represent.
- The historical-support audit of the 200 most confident zero-detection cases
  shows two comparably important mechanisms. Eighty cases (40%) had prior
  same-season support but had never detected the species, indicating
  ecological/locality overgeneralization or missing habitat predictors.
  Seventy-six (38%) had previously detected the species in the same season,
  and another six had prior detections in other seasons, indicating temporal
  availability change or observer/reporting nonstationarity. Nineteen (9.5%)
  were genuinely unsupported localities. The remaining 19 had prior locality
  support but no historical detection in any season.
- Personal locations account for 128 of the 200 cases and generally have one
  observer, but 72 cases are hotspots with many observers. Both historical
  failure classes occur in both locality types, so observer identity alone
  cannot explain the pattern.
- Continue the repeated-visit path with guardrails. The general framework
  should preserve a portable ecology-based availability model for unseen
  localities and may later add partially pooled locality/observer dynamics for
  revisited sites, falling back to the portable component when history is
  absent. Do not add a GNN to `psi` yet.
- The next diagnostic evaluates all held-out focus-species pairs, not only the
  selected failures, across seen/unseen localities, recent same-season
  detection history, observer diversity, locality type, and season. This is
  needed before deciding whether to implement the dynamic locality/observer
  layer or move directly to explicit locality-held-out transfer training.
- The full transfer-strata diagnostic is now complete across 45,640 held-out
  focus-species/locality-season pairs. Seen localities are well calibrated in
  aggregate (observed/predicted any-detection 0.3609/0.3632, signed error
  +0.0023), while naturally unseen localities are overpredicted
  (0.2865/0.3350, signed error +0.0486). Macro AUPRC falls from 0.6293 at seen
  localities to 0.5087 at unseen localities, and mean absolute species
  calibration error rises from 0.0157 to 0.0647. The portable component
  therefore has a real locality-transfer gap that pooled temporal metrics hide.
- Historical state is independently important within previously sampled
  localities. Species detected in the latest prior same-season year are
  strongly underpredicted (observed/predicted 0.8231/0.6376, signed error
  -0.1855), while species never detected despite prior same-season sampling are
  strongly overpredicted (0.0879/0.1985, signed error +0.1106). The
  past-detection/recent-zero stratum is nearly calibrated in aggregate
  (-0.0067) but has high BCE (0.6709), so its pooled mean conceals difficult
  case-level transitions.
- Effort and locality type do not explain the pattern alone. One-observer
  groups are overpredicted by +0.0517 while 6+ observer groups are
  underpredicted by -0.0400; personal locations are overpredicted by +0.0508
  while hotspots are underpredicted by -0.0241. These opposing errors support
  an ecological/locality-history interpretation rather than a single global
  observer-density correction.
- Both mechanisms matter, but portability must be measured before adding an
  adaptive history layer. The next fixed-likelihood run therefore uses a
  controlled temporal-locality split: choose representative established
  localities using only non-outcome geography, environment, replication,
  effort-diversity, season, and locality-type balance; remove all of their
  pre-2023 groups from training; and evaluate only their 2023 groups. Historical
  detections remain unavailable to the portable fit. A later shared history
  component may adapt predictions at revisited localities while falling back
  to the portable model when history is unavailable; species deviations are a
  later sensitivity only if the shared effect underfits consistently.
- The controlled temporal-locality run
  (`latent_strict_frailty_global_localityxfer_s37`) is complete. It trained on
  9,650 groups from non-held-out pre-2023 localities and tested 706 2023 groups
  from 269 held-out established localities (17,827 checklists), with no
  locality overlap. The selected test set represented 19.8% of eligible
  test-year groups and retained a low non-outcome balance score of 0.0335.
- Portable ranking transfers well. Checklist micro AUROC is 0.87117 versus
  0.87018 under the broader temporal test, while mean species AUROC is
  0.80962 versus 0.80923. Raw micro AUPRC rises from 0.57279 to 0.59619 because
  controlled-holdout prevalence is higher (0.17509 versus 0.16115); AUPRC lift
  over prevalence is slightly lower (3.405 versus 3.554), and mean
  species-level AUPRC lift is essentially unchanged (3.723 versus 3.731).
  Therefore the higher raw AUPRC is not evidence of model improvement, but the
  stable normalized ranking is evidence against catastrophic locality
  memorization.
- Probability transfer is weaker than ranking transfer. Checklist marginal
  calibration error increases from 0.00888 to 0.01583, ECE from 0.01337 to
  0.01972, and mean absolute species calibration error from 0.01049 to 0.02359.
  Fair group any-detection changes from a +0.00220 signed error to -0.01442.
  The controlled model underpredicts 6+ observer groups by -0.05359 and
  overpredicts one-observer groups by +0.04137, reproducing the opposing
  support pattern seen in the earlier transfer audit.
- Detection dependence also transfers imperfectly: weighted pairwise
  co-detection error changes from -0.01562 to -0.02477, with the largest
  controlled-holdout error in one-observer groups (-0.04115). The fitted global
  frailty scale is nevertheless almost identical (1.07715 versus 1.07439), so
  this does not justify reopening the frailty-variance sweep. It instead points
  to locality/observer heterogeneity not represented by a single global scale.
- Treat the controlled run as a qualified portable-baseline pass. The next
  diagnostic applies the same historical-state strata to these held-out
  localities. All are unseen by the fitted model, but their pre-2023 records
  remain available for diagnosis and for a later adaptive model. This cleanly
  separates portable prediction from the potential gain of history-aware
  adaptation before either component receives graph structure.
- The controlled transfer-strata diagnostic is now complete. Across all 7,060
  held-out focus-species pairs, observed and predicted any-detection rates are
  almost identical (0.3636 versus 0.3634; signed error -0.0002), but this pooled
  result again hides large, opposed history errors. Latest-prior-year
  detections are underpredicted by -0.1849 (0.8163 observed versus 0.6314
  predicted), while species never detected despite prior same-season sampling
  are overpredicted by +0.1058 (0.0951 versus 0.2009). These are nearly the
  same errors as in the broader temporal test (-0.1855 and +0.1106).
- The historical-state result is not driven by one or two focus species. All
  ten focus species have a negative signed error after a latest-prior-year
  detection and a positive signed error in the never-detected same-season
  stratum. The controlled split therefore turns prior same-season history into
  a supported transferable predictor, rather than a post-hoc locality patch or
  evidence of training leakage.
- Opposing observer/locality errors also reproduce: one-observer and personal
  locations are overpredicted by +0.0599 and +0.0579, while 6+ observer and
  hotspot groups are underpredicted by -0.0470 and -0.0307. These should remain
  diagnostics; adding a single observer-density offset would average over the
  problem rather than explain it.
- Implement the first adaptive availability bridge as a deliberately small
  shared history correction, not a species-specific lookup. For each target
  locality/species/biological-season, it uses only earlier years and encodes
  three mutually exclusive states: detected in the latest prior year, sampled
  but never detected in prior same-season years, and detected historically but
  zero in the latest prior year. Each state is attenuated by bounded prior
  checklist support. No prior same-season support produces an exactly zero
  history-logit correction, preserving the model's portable ecological
  fallback structurally.
- Save both the portable and history-adapted availability predictions. The
  transfer diagnostic compares both surfaces on identical held-out pairs, so
  the history term is promoted only if it reduces the two demonstrated
  history-state errors without degrading no-history transfer, fair checklist
  calibration, phenology, or environmental-response plausibility. The base
  coefficients are still estimated jointly in this first test, so "zero
  fallback" means no direct history correction; it does not claim numerical
  identity to the separately fitted non-history checkpoint.
- The shared adaptive-history run
  (`latent_xfer_s37_histshared_l2_0p01`) is complete on the identical seed-37
  controlled locality split. Relative to the controlled no-history fit,
  checklist micro AUROC improved from 0.87117 to 0.88239, micro AUPRC from
  0.59619 to 0.60996, and BCE from 0.31874 to 0.30811. Mean-rate calibration
  error improved from 0.01583 to 0.01301, while ECE changed slightly from
  0.01972 to 0.02077. The global frailty scale remained stable at 1.07325, so
  the gain is attributable to history structure rather than a changed
  repeated-detection variance estimate.
- The within-run portable ablation confirms that the history term addresses its
  intended target. For latest-prior-year detections, signed any-detection error
  improved from -0.2076 to -0.1225 and BCE from 0.5446 to 0.4463. For prior
  same-season sampling with no detections, signed error improved from +0.1042
  to +0.0557 and BCE from 0.3166 to 0.2823. Nine of ten focus species improved
  in absolute calibration in the first stratum and all ten improved in the
  second. No-prior-same-season predictions were identical, verifying the zero
  direct-correction fallback.
- Learned shared history-logit weights are coherent: +0.94448 for detection in
  the latest prior same-season year, -0.73091 for prior same-season sampling
  that never detected the species, and +0.04576 for an older detection followed
  by a latest-year zero. The near-zero third weight is useful evidence that the
  model is not treating every historical detection as persistent occupancy.
- Focus-species transfer improved materially: within-run portable versus
  adapted AUPRC is 0.8222 versus 0.8642, macro AUPRC 0.6154 versus 0.7187,
  BCE 0.4201 versus 0.3693, and mean absolute species calibration error 0.0348
  versus 0.0279. The adapted pooled signed error is -0.0080.
- Important limitations remain. Across all 100 species, fair group
  any-detection underprediction worsened from -0.01442 in the independently
  fitted no-history model to -0.02281 in the adaptive model, although weighted
  pairwise co-detection underprediction improved from -0.02477 to -0.02125.
  The two targeted history errors are reduced but remain substantial, and
  House Sparrow remains overpredicted in every season. High-confidence
  zero-detection cases remain for Northern Cardinal and Eastern Towhee, which
  is expected when a latest prior detection is followed by a real state
  transition that the static history summary cannot foresee.
- Treat the shared history component as the promoted adaptive branch for
  previously sampled locality/species/seasons, while retaining the controlled
  no-history model as the explicit portable branch. Do not add species-specific
  history deviations: the shared direction is consistent and the added
  complexity is not currently justified. Before broader transfer tests, run
  the same availability diagnostic on the controlled no-history checkpoint for
  a direct phenology, environmental-response, and high-support comparison.
- The controlled apples-to-apples availability comparison is complete and is a
  qualified plausibility pass for shared history. Relative to the independently
  fitted no-history checkpoint, the adaptive run reduced the largest
  high-support signed errors for House Sparrow (+0.1413 to +0.1266),
  Double-crested Cormorant (-0.1263 to -0.1130), Wood Thrush (-0.0812 to
  -0.0722), Red-headed Woodpecker (-0.0712 to -0.0599), and Eastern Towhee
  (-0.0482 to -0.0337). Smaller near-zero errors for Eastern Meadowlark, Great
  Egret, Northern Cardinal, and Black-and-white Warbler moved modestly in the
  wrong direction, so the result is not uniformly better.
- Environmental-response plausibility improved rather than trading off against
  the history gain. Mean weighted observable MAE fell for canopy (0.0437 to
  0.0419), coastline distance (0.0451 to 0.0410), waterbody distance (0.0483 to
  0.0435), and elevation (0.0481 to 0.0455). Observable-response shape
  Spearman increased from 0.67/0.86/0.72/0.82 to
  0.76/0.88/0.80/0.89 respectively; latent-availability shape was retained or
  improved for all four covariates.
- Broad phenology remains recognizable, but the comparison is mixed at the
  species-season level. House Sparrow and Eastern Towhee errors improved;
  Wood Thrush was broadly stable with one improved and two nearly unchanged
  migration/breeding bins; Green Heron, Eastern Meadowlark, and
  Double-crested Cormorant worsened modestly in several bins. This is a reason
  to retain the portable surface and repeat the test in another year, not a
  reason to add species-specific history coefficients.
- High-confidence zero-detection behavior improved on average for most focus
  species, including lower mean prior any-detection probabilities for Northern
  Cardinal, Double-crested Cormorant, Eastern Towhee, Great Egret, Green
  Heron, Wood Thrush, and Red-headed Woodpecker. A few upper-tail cases became
  more extreme, notably for Eastern Towhee, House Sparrow, Green Heron,
  Black-and-white Warbler, and Eastern Meadowlark. Shared history therefore
  reduces the dominant systematic bias but does not solve abrupt state changes
  or every locality-specific failure.
- Advance to a paired 2022 temporal-locality replication with all likelihood,
  support, split-selection, and history hyperparameters fixed. Fit both the
  portable no-history model and the shared-history model on the same seed-37
  split. The separately fitted portable run is required because the adaptive
  run's zero-history ablation shares jointly estimated base coefficients and is
  not an independent baseline.
- The independently fitted 2022 portable checkpoint
  (`latent_xfer22_s37_nohist`) is complete. It trained on 5,885 pre-2022
  groups and tested 633 groups from 235 held-out established localities
  (16,097 checklists), with no locality overlap. The test group fraction was
  0.198 and the non-outcome split-balance score was 0.0446, so the controlled
  design remains acceptably representative despite the smaller earlier-year
  training set.
- Raw checklist AUPRC declined from the 2023 portable result's 0.59619 to
  0.54669, but prevalence also declined from 0.17509 to 0.15885. AUPRC lift
  over prevalence is therefore 3.4415 versus 3.405 in 2023, while AUROC is
  0.85787 versus 0.87117. This is a qualified ranking-transfer pass rather
  than evidence of a major deterioration.
- Probability and dependence diagnostics are stronger in 2022. Checklist
  mean-rate error is 0.00231 and ECE 0.00486, versus 0.01583 and 0.01972 in
  the 2023 portable run. Fair group any-detection remains mildly
  underpredicted (-0.01756 versus -0.01442), while weighted pairwise
  co-detection underprediction improves to -0.00606 from -0.02477. The global
  frailty scale is effectively unchanged at 1.07885 versus 1.07715.
- The same observer-support structure reproduces: one-observer groups are
  overpredicted by +0.04548 and 6+ observer groups are underpredicted by
  -0.06273, close to +0.04137 and -0.05359 in 2023. Large Pine Siskin and
  Red-breasted Nuthatch winter/seasonal errors expose interannual ecological
  dynamics rather than a failure of the controlled split. Proceed with the
  fixed 2022 shared-history run; do not retune the likelihood or split.
- The fixed 2022 shared-history checkpoint
  (`latent_xfer22_s37_histshared_l2_0p01`) is complete on the identical 633
  groups and 235 held-out localities. Relative to the independent 2022
  portable fit, checklist AUROC improved from 0.85787 to 0.86844, AUPRC from
  0.54669 to 0.55758, and BCE from 0.31219 to 0.30391. Mean-rate error improved
  from 0.00231 to 0.00116 and max-bin error from 0.03224 to 0.02378; ECE
  increased slightly from 0.00486 to 0.00647 but remains small.
- The 2023 tradeoff also replicated. Fair all-species group any-detection
  underprediction worsened from -0.01756 to -0.02522, while weighted pairwise
  co-detection error improved from -0.00606 to -0.00137. The global frailty
  scale remained effectively fixed (1.07885 versus 1.07660), again attributing
  the change to availability history rather than detection dependence.
- Observer-support errors mostly narrowed: one-observer error improved from
  +0.04548 to +0.02692, two-observer error from +0.02670 to +0.00850, and 6+
  observer error from -0.06273 to -0.05984; the 3-5 observer stratum worsened
  from -0.01726 to -0.03169. Several negative species-season checklist errors
  improved, but Pine Siskin and Red-breasted Nuthatch overprediction worsened.
  Treat this as a replicated aggregate adaptive gain with a replicated
  group-calibration cost, not final promotion evidence. Run the history-strata
  comparison next to verify that the intended historical states improved and
  that no-history fallback remained unchanged.
- The paired 2022 transfer-strata comparison is complete and confirms
  mechanistic replication. Within the adaptive fit, latest-prior-year
  any-detection error improved from -0.2146 to -0.1375, BCE from 0.5423 to
  0.4394, and mean absolute species calibration error from 0.2906 to 0.1942.
  Prior same-season sampling with no detections improved from +0.0812 to
  +0.0518, BCE from 0.3214 to 0.3056, and species calibration error from
  0.0978 to 0.0680.
- No-prior-same-season predictions were exactly unchanged within the adaptive
  fit, confirming the structural portable fallback in a second held-out year.
  The past-detection/recent-zero stratum was also unchanged, consistent with
  the near-zero weight learned for that ambiguous transition in 2023 rather
  than indiscriminate persistence of every historical detection.
- Relative to the independently fitted 2022 portable checkpoint, adapted
  latest-prior-year error improved from -0.1928 to -0.1375 and
  never-detected error from +0.0877 to +0.0518. Overall focus-species transfer
  AUPRC improved from 0.8347 to 0.8675, macro AUPRC from 0.6523 to 0.7503,
  and BCE from 0.4152 to 0.3757. Pooled signed error changed slightly from
  -0.0132 to -0.0160 and species calibration MAE from 0.0215 to 0.0222, so
  ranking and subgroup corrections improved without eliminating aggregate
  calibration tension.
- Observer and locality-type behavior also moved in the intended direction:
  one-observer error improved from +0.0581 to +0.0438, 2-5 observers from
  +0.0134 to +0.0024, 6+ observers from -0.0713 to -0.0629, hotspots from
  -0.0497 to -0.0463, and personal locations from +0.0588 to +0.0438.
  Shared history is therefore promoted as a replicated adaptive component for
  revisited locality/species/seasons, with the independent no-history surface
  retained for portability. The remaining 2022 gate is availability
  plausibility, especially the year-sensitive species and high-confidence
  zero-detection tails.
- The paired 2022 availability diagnostic is complete and is another qualified
  pass. Shared history reduced mean weighted observable-response MAE for
  canopy (0.0456 to 0.0433), coastline distance (0.0445 to 0.0403),
  waterbody distance (0.0389 to 0.0368), and elevation (0.0494 to 0.0447).
  Observable-response shape improved for canopy and coastline, was unchanged
  for waterbody distance, and declined slightly for elevation (0.76 to 0.74);
  latent-availability shape improved for canopy, coastline, and elevation and
  was unchanged for waterbody distance.
- Eight of ten focus species improved their absolute high-support observable
  error. The largest gains were Double-crested Cormorant (-0.1499 to
  -0.1112), Eastern Towhee (-0.1370 to -0.1039), Eastern Meadowlark (-0.1126
  to -0.0932), Red-headed Woodpecker (-0.1077 to -0.0918), and Northern
  Cardinal (-0.0586 to -0.0400). Wood Thrush and Black-and-white Warbler
  worsened modestly. Broad seasonal patterns remained recognizable, with mixed
  species-bin changes rather than a systematic phenology collapse.
- Mean predicted any-detection probability among high-support zero-detection
  groups fell for all ten focus species. Several maxima nevertheless increased,
  including Eastern Towhee, Double-crested Cormorant, Wood Thrush,
  Black-and-white Warbler, House Sparrow, and Green Heron. This repeats the
  2023 conclusion: shared history reduces systematic error but redistributes
  some abrupt-transition risk into a smaller extreme tail.
- The paired 2022 gate is passed. Retain a two-surface framework: the
  independent ecology/season model is the portable estimate for unsupported
  locality/species/seasons, and shared history is the adaptive estimate where
  earlier same-season records exist. Do not add species-specific history or
  post-hoc calibration. Proceed to directional ecological-regime transfer with
  both surfaces and the likelihood fixed.
- A generic leakage-safe `temporal-regime` split is now implemented for that
  transfer phase. It derives a locality's regime value only from pre-test-year
  groups, holds an inclusive low or high tail of established localities out of
  all training years, and tests those localities only in the requested year.
  The first test is the low historical distance-to-coastline tail. This is a
  directional extrapolation stress test, not a representative benchmark or an
  NC-specific ecological-region definition.
- The portable 2023 coastal-tail stress test is complete. It held out 269 of
  1,343 established localities at an inclusive pre-2023 median coastline-
  distance threshold of 4,852 EPSG:3857 map meters, representing 20.6% of eligible test groups,
  with zero held-out locality overlap in training.
- Relative to the representative seed-37 locality holdout, checklist AUROC fell
  from 0.8712 to 0.8021 and prevalence-normalized AUPRC from 3.405 to 2.916.
  Availability any-positive AUROC fell from 0.8528 to 0.8071, while its AUPRC
  lift was more stable at 1.847 versus 1.820. Fair group any-detection error
  widened from -0.0144 to -0.0499, checklist ECE rose from 0.0197 to 0.0255,
  and maximum bin error rose from 0.0624 to 0.1424.
- The loss is also species-level, not only a pooled prevalence effect.
  Checklist macro AUROC/AUPRC fell from 0.8096/0.4214 to 0.7542/0.2690,
  mean absolute species calibration error rose from 0.0236 to 0.0497, and
  availability macro AUROC/AUPRC fell from 0.7769/0.6465 to 0.6943/0.5372.
- The global frailty scale remained stable at 1.0715 and weighted pairwise
  co-detection error narrowed to -0.0013. The largest remaining errors are
  concentrated in coastal specialists, especially Laughing Gull,
  Boat-tailed Grackle, Brown Pelican, and Royal Tern. This is a real portable
  ecological-regime transfer gap rather than evidence that the repeated-visit
  likelihood or frailty component failed. Run the matching shared-history
  checkpoint to measure recoverable adaptation separately from portability.
- The matching shared-history coastal run is complete. Relative to the
  independently fitted portable coastal model, checklist AUROC/AUPRC improved
  from 0.8021/0.3626 to 0.8295/0.3950, BCE improved from 0.3120 to 0.2951,
  and ECE improved from 0.0255 to 0.0199. Checklist macro AUROC/AUPRC improved
  from 0.7542/0.2690 to 0.7973/0.3076, and mean absolute species calibration
  error improved from 0.0497 to 0.0456. Checklist AUPRC improved for 93 of 100
  species.
- Availability ranking improved more strongly: pooled any-positive
  AUROC/AUPRC rose from 0.8071/0.7035 to 0.8618/0.7841, and macro
  AUROC/AUPRC rose from 0.6943/0.5372 to 0.7980/0.6735. Mean absolute
  species availability error against the observed-positive lower bound fell
  from 0.0716 to 0.0627. The learned shared logit weights were coherent:
  +0.9346 for a latest-prior-year detection, -0.6898 for never detected in
  the same season, and +0.0396 for a past detection followed by a recent zero.
- The adaptive gain does not close the coastal transfer problem. Fair group
  any-detection error widened slightly from -0.0499 to -0.0547 and maximum
  checklist-bin error changed from 0.1424 to 0.1437. Large negative errors
  remain for Laughing Gull, Boat-tailed Grackle, Brown Pelican, and other
  coastal specialists, although many are smaller than in the portable fit.
  Frailty remained stable at 1.0676 and weighted pairwise co-detection error
  remained near zero at +0.0005. Shared history is therefore useful adaptive
  information at revisited coastal localities, not a substitute for portable
  coastal habitat representation.
- A preprocessing audit found no evidence that barrier-island observations
  were simply omitted or assigned distance to the mainland/state boundary.
  The coastline band is built from the USGS Small-scale 1:1,000,000 Coastline
  layer, and direct raster samples near Emerald Isle, Fort Macon, and Ocracoke
  had coastline distances of approximately 660 m, 30 m, and 812 m. The final
  checklist table has 661,979 rows with no missing values in the four retained
  raster covariates; 114,820 rows are within the 4,852-map-meter coastal-tail
  threshold. The selected tail is geographically confined to eastern NC.
- The same audit identified important limitations. The distance raster is on
  an EPSG:3857 template, so values called meters are Web-Mercator map meters
  and are inflated at NC latitude; this affects physical interpretation more
  than rank order. More importantly, the generalized Waterbody layer gives
  multi-kilometer waterbody distances at barrier-island examples and does not
  adequately encode ocean, sound, estuary, salt-marsh, dune, or barrier-island
  habitat. Canopy gaps were also filled before stacking, and
  `--drop-missing-raster-covariates any` makes the raster footprint an implicit
  sample-selection filter without preserving stage-specific retention counts.
  The README additionally used the obsolete `--valid-footprint intersection`
  option; the current stack script uses `--extent intersection`.
- Accordingly, retain the run as a valid **generalized-coastline proximity
  stress test**, but do not describe it as a complete coastal-habitat transfer
  test. The evidence is inconsistent with the failure being only a malformed
  eBird preprocessing artifact. It is consistent with a combination of real
  extrapolation, sparse coastal analogs, and under-specified coastal ecological
  covariates. Before drawing stronger coastal conclusions, audit raw-to-final
  retention by coastal distance and rebuild a sensitivity covariate stack in
  an appropriate metric CRS with explicit ocean/sound, estuarine/tidal
  wetland, land-cover, and barrier-island context. Do not overwrite the current
  dataset; preserve it as the coarse-covariate baseline.

- The paired coastal diagnostics reinforce that interpretation. Shared history
  improved unseen-locality focus-pair AUPRC from 0.7692 to 0.8263 and macro
  AUPRC from 0.5505 to 0.6825. It reduced the latest-prior-detection signed
  error from -0.2855 to -0.2027 and the never-detected-same-season error from
  +0.0955 to +0.0463. Ranking also improved in every observer-diversity and
  locality-type stratum. These are coherent adaptive-history gains.
- History still does not solve portable coastal extrapolation. Unseen-locality
  mean prediction remained low by 0.0568, hotspot localities remained low by
  0.0692, and some species moved in the wrong direction. The environmental
  response audit improved mean observable MAE for canopy, coastline distance,
  waterbody distance, and elevation from 0.0885/0.0848/0.0891/0.0952 to
  0.0815/0.0778/0.0803/0.0848, but absolute errors remain much larger than on
  representative locality transfer. Shared history should remain the adaptive
  branch; it is not a replacement for richer portable ecological covariates.
- The next data change should therefore be a versioned national covariate
  feature pipeline, first piloted in NC and then reused without state-specific
  logic. Authoritative national source layers should supply the raw geography;
  this project should derive consistent point and multi-scale neighborhood
  features from them. NC-specific hand labels are not the intended solution.

### Current Promoted Latent Model

For species `j`, locality-season-year group `g`, and checklist visit `i` within
that group, the model has a latent availability state

$$
z_{jg} \sim \operatorname{Bernoulli}(\psi_{jg}).
$$

The portable availability surface is a species-specific logistic regression:

$$
\operatorname{logit}(\psi^{(0)}_{jg})
  = \alpha^{\psi}_j + x_g^\mathsf{T}\beta^{\psi}_j,
$$

where `x_g` contains the biological-season indicators, season year, canopy,
elevation, log distance to the generalized waterbody layer, and log distance
to the generalized coastline layer. Coordinates are not included in the
promoted runs. The adaptive surface adds a small shared history correction:

$$
\operatorname{logit}(\psi^{(H)}_{jg})
  = \operatorname{logit}(\psi^{(0)}_{jg}) + h_{jg}^\mathsf{T}\gamma.
$$

The three mutually exclusive entries of `h_jg` encode latest-prior-year
detection, never detected in prior same-season visits, and a past detection
followed by a recent zero. They use only earlier years and are attenuated by
`1 - exp(-n_prior_checklists / 20)`; all are exactly zero when no prior
same-season history exists. `gamma` is shared across species. The portable and
adaptive estimates are separately fitted models; the adaptive model's
zero-history ablation is not used as a replacement for the independent
portable checkpoint.

Conditional detection is also species-specific:

$$
\operatorname{logit}(p_{ijg}(u))
  = \alpha^p_j + w_i^\mathsf{T}\beta^p_j
    + \delta^p_{j,s(g)} + \sigma u_{jg},
\qquad u_{jg} \sim \mathcal{N}(0,1).
$$

The checklist vector `w_i` contains log duration, log travel distance, log
observer count, stationary/traveling indicators, and cyclic day-of-year,
day-of-week, and start-time terms. `delta` is the L2-regularized
species-by-biological-season detection offset. The promoted frailty uses one
shared scale `sigma` across species but integrates a latent detection
propensity within each locality-season/species group; the coastal adaptive run
estimated `sigma = 1.0676` logit units. Seven-point Gauss-Hermite quadrature is
used for this integral.

Given availability and frailty, checklist outcomes follow

$$
y_{ijg} \mid z_{jg},u_{jg}
  \sim \operatorname{Bernoulli}\left(z_{jg}p_{ijg}(u_{jg})\right).
$$

For a group with at least one detection, its species likelihood is

$$
L_{jg}
  = \psi_{jg}\int \phi(u)
      \prod_{i\in g}p_{ijg}(u)^{y_{ijg}}
      [1-p_{ijg}(u)]^{1-y_{ijg}}\,du.
$$

For an all-zero group, the likelihood retains both biological absence and
present-but-missed explanations:

$$
L_{jg}
  = (1-\psi_{jg})
    + \psi_{jg}\int \phi(u)\prod_{i\in g}[1-p_{ijg}(u)]\,du.
$$

The fair prior-marginal checklist prediction is
`P(y_ijg = 1) = psi_jg * E_u[p_ijg(u)]`. The fair group any-detection prediction
is `psi_jg * {1 - E_u[product_i(1 - p_ijg(u))]}`. Training minimizes mean
negative log likelihood with pooled and species-level marginal-rate penalties
(weights 25 and 10), species-season L2 0.0025, frailty L2 0.01, adaptive-history
L2 0.01 when enabled, and AdamW weight decay 0.0001.

This current model is deliberately **not a GNN**. It is the structural,
interpretable baseline against which a later availability-side GNN must show
transferable ecological gain after effort, repeated visits, history, and
within-group detection dependence have already been represented.

This is still the right path, but success is no longer defined as squeezing a
few additional points of checklist AUPRC from the NC testbed. The purpose of
the latent branch is to estimate a stable, interpretable availability surface
while preserving a separately testable observation process. The bridge remains
ahead on raw checklist prediction, which is acceptable unless component
diagnostics show that the latent separation is unstable or biologically
implausible.

### National Covariate Enrichment Strategy

There is no single national bird-habitat dataset that should replace the
current four ecological predictors. The portable design is a **national core
plus optional regional modules**, with all derived features generated by one
versioned pipeline.

National ecological core:

- [USGS Annual NLCD](https://www.usgs.gov/centers/eros/science/about-annual-nlcd)
  provides annual 30 m CONUS land cover and fractional imperviousness from
  1985 through 2025. Derive class fractions, imperviousness, fragmentation,
  and recent land-cover change at multiple neighborhood scales rather than
  using only the class at the checklist point.
- [LANDFIRE existing vegetation](https://landfire.gov/vegetation) provides
  30 m Existing Vegetation Type, Cover, and Height products with full-extent
  CONUS downloads. These add vegetation composition and vertical structure
  that canopy percentage alone cannot represent.
- [USGS 3DEP](https://www.usgs.gov/3d-elevation-program) is the national
  elevation source. Derive elevation, slope, transformed aspect, topographic
  position, relief, and terrain heterogeneity in an equal-area metric CRS.
- [USGS 3DHP](https://www.usgs.gov/3d-hydrography-program/access-3dhp-data-products)
  is replacing NHD, WBD, and NHDPlus HR. During the transition, retain a
  versioned fallback to legacy NHDPlus HR where 3DHP coverage is incomplete.
  Derive separate stream, river, lake/reservoir, and waterbody distances,
  densities, and neighborhood fractions instead of one generalized water
  distance.
- [Daymet](https://daymet.ornl.gov/) provides daily 1 km temperature,
  precipitation, vapor pressure, radiation, snow-water equivalent, and day
  length across continental North America. Derive season-matched normals,
  anomalies, heat/cold, and precipitation summaries; these are ecological
  availability predictors, not checklist effort variables.
- [USFWS National Wetlands Inventory](https://www.fws.gov/program/national-wetlands-inventory/wetlands-data)
  provides Cowardin-classified wetland and deepwater polygons by state or HUC8.
  Derive wetland-system fractions and distances, but preserve source-project
  date and coverage metadata because NWI vintages vary and are updated
  incrementally.

Coastal module:

- [NOAA C-CAP Regional Land Cover](https://www.coast.noaa.gov/digitalcoast/data/ccapregional.html)
  supplies nationally consistent 30 m coastal land cover, including coastal
  wetland classes, for U.S. coastal zones. It is a regional supplement, not a
  CONUS core predictor.
- [NOAA CUSP shoreline](https://nsde.ngs.noaa.gov/) supplies a contemporary
  national shoreline. Combine it with NWI/C-CAP and hydrography to distinguish
  ocean shoreline, estuary/sound, tidal wetland, freshwater, and inland water.
- For barrier-island and narrow-coastal-land contexts, derive neutral geometry
  features rather than an NC-specific island flag: distance to ocean-facing
  shoreline, distance to back-barrier/estuarine water, surrounding open-water
  fraction, local land-strip width, tidal-wetland fraction, and dune/barren or
  beach land-cover fraction where the source classification supports it.

Evaluation and observation-process layers:

- [EPA ecoregions](https://www.epa.gov/eco-research/ecoregions) should first be
  used to construct and report transfer splits, not as a shortcut predictor.
- [USGS PAD-US](https://www.usgs.gov/programs/gap-analysis-project/science/pad-us-data-download),
  Census population/geographies, and
  [TIGER/Line roads](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)
  can represent public access, urbanization, and travel accessibility. Keep
  these in the effort/observer-access component by default so the availability
  model is not rewarded for reproducing where people bird.

Implementation rules for portability:

1. Use one analysis grid in CONUS Albers (`EPSG:5070`) for metric distances and
   areas. Preserve source rasters/vectors and their native metadata separately.
2. Do not create a monolithic native-resolution CONUS raster stack. Process by
   state/HUC/tile and write a compact locality/checklist feature table.
3. Derive the same point and buffered summaries everywhere. Initial scales
   should be approximately 250 m, 1 km, and 5 km, with source-specific tests
   before adding more scales.
4. Match dynamic layers to the observation year when available. Store source
   name, version, acquisition/map year, coverage status, and distance from the
   nearest valid source observation as model-auditable fields.
5. Keep the existing four-variable dataset unchanged as the coarse baseline.
   Build an enriched NC sensitivity dataset beside it and change covariates
   while holding the latent likelihood, split, support rules, and optimizer
   fixed.
6. Add sources in interpretable blocks and ablate them: land cover/vegetation,
   wetland/hydrography, terrain/climate, then the coastal supplement. A source
   is promoted only if it improves transfer across species and regimes without
   degrading phenology, environmental plausibility, and fair calibration.
7. After the NC pilot, validate the same feature definitions in at least one
   additional region before a full CONUS build. State-specific data may be
   used for auditing but not as required model input.

## Historical Analysis Checkpoints

This document is both the methods plan and the analysis ledger. Older sections
are intentionally retained as a timeline of what was tested, including negative
results. The current working interpretation is:

- Complete-checklist eBird is not pure presence-only data. The retained data
  contain checklist-level detections and informative non-detections under an
  observation process.
- Checklist-level tabular and graph models remain useful for modeling
  detection/reporting, but they do not by themselves fully separate biological
  availability from observer effort.
- The strongest repeated checklist-level result is that a portable
  ecological/temporal/effort detector explains most of the signal. Spatial GNN
  residuals add modest, sometimes useful improvement, but architecture variants
  alone have mostly moved metrics within a narrow band.
- The current preferred checklist-level GNN benchmark is
  `spatial_gcn_frozen_access_h64_l2_z64`. It is a benchmark residual framework,
  not the final scientific model.
- Support-gate variants are documented as negative or mixed results. A
  cell-density gate improved pooled calibration slightly but reduced ranking;
  the species-cell gate broadly degraded ranking and calibration. These results
  argue against naive support-based suppression as the main path.
- The methodological pivot is to use repeated complete checklists at the same
  localities through biological seasons. This locality-season view is the
  in-dataset route toward separating ecological availability/occupancy from
  checklist-level detection.
- The active next track is therefore:
  `locality-season availability component + checklist detection component`.
  The completed binomial locality-season baseline is a bridge toward that
  target, not a full latent occupancy model yet.
- The first full locality-season baseline supports the pivot. The combined
  availability + effort model has the best held-out weighted BCE and
  positive-triplet ranking, availability-only is clearly stronger than
  effort-only, and effort summaries add broad complementary signal when combined
  with availability features.
- The next model should not be judged only by whether pooled AUROC/AUPRC moves a
  few points. The key question is whether the availability component preserves
  biologically plausible seasonal/ecological structure while the checklist-level
  component explains effort-driven detection variation.
- The first two-component checklist detection run supports the pivot. It
  improved held-out checklist-level BCE and micro/species AUPRC over
  availability-only and effort-only while preserving key focus-species
  phenology checks such as Wood Thrush winter near-zero detection.
- The active diagnostic question is now calibration heterogeneity and
  ecological plausibility, not whether the two-component bridge has detection
  signal. Filtered probability-bin diagnostics show that tiny-bin artifacts
  should be ignored and that the stable remaining issues are county-season and
  focus-species season calibration pockets. The next step should be response
  and phenology diagnostics before another model architecture change.
- The updated 10/20 two-component run generated the new focus-species monthly
  phenology and environmental-response diagnostics. Initial review suggests the
  two-component correction usually preserves ecological response ordering but
  sometimes worsens probability levels relative to availability-only. The next
  step is visual component inspection followed by targeted detection-component
  shrinkage if the same pattern is confirmed.
- The first combined neutral-point shrinkage run (`residual_l2=0.01`,
  `availability_weight_l2=0.01`) is a mixed result. It materially improves pooled
  and mean-species calibration with only a small ranking loss, but aggregate
  monthly phenology error worsens slightly and several species lose more AUPRC
  than the pooled change suggests. The penalties should therefore be ablated
  separately before adopting regularization as the preferred model.
- The residual-only `0.01` ablation is almost identical to the combined run but
  slightly worse on pooled AUPRC, monthly MAE, and max-bin calibration. This
  indicates that residual/effort shrinkage drives most of the broad calibration
  change and most of the ranking cost.
- The availability-weight-only `0.01` ablation is effectively neutral. It
  preserves pooled and species ranking, phenology, response error, and
  calibration almost exactly. Therefore availability-weight shrinkage alone
  does not solve the calibration issue, and its small contribution in the
  combined run is secondary to residual shrinkage.
- Residual-only shrinkage at `0.0025` produced a real intermediate tradeoff. It
  retained about 62% of the `0.01` ECE improvement while incurring about 47% of
  the pooled AUPRC loss, 45% of the mean-species AUPRC loss, and 54% of the
  focus-species phenology degradation. It also retained nearly all of the small
  environmental-response MAE improvement.
- The final midpoint residual L2 `0.005` run confirmed a smooth monotonic
  ranking/calibration tradeoff rather than a sharp optimum. It retained about
  81% of the `0.01` ECE improvement but incurred about 73% of its pooled AUPRC
  loss, 71% of its mean-species AUPRC loss, and 75% of its phenology cost.
- Scalar L2 tuning is now closed. Keep the unregularized model as the ranking
  reference and use `0.0025` as the conservative regularized sensitivity model:
  it provides a meaningful calibration improvement with less ecological and
  species-ranking disruption than `0.005` or `0.01`.
- The first partial-pooling effort-response run is a safe but not superior
  variant. It behaves almost identically to weak species-specific residual
  shrinkage: a small calibration improvement relative to the unregularized
  model, a small ranking cost, and no evidence yet that partial pooling solves
  the remaining species/regime issues.
- The fully shared effort-response ablation is a negative result. It is still
  better than availability-only, but it loses substantial checklist-level and
  species-level ranking relative to all species-specific or partially pooled
  effort models, and it does not improve pooled calibration. This indicates
  that species-specific detectability responses are necessary; a single shared
  effort effect is too restrictive.
- Effort-mode testing is now closed for the current two-component bridge. Keep
  the unregularized species-specific model as the ranking reference, keep
  species-specific residual L2 `0.0025` as the conservative regularized
  sensitivity model, and treat partial pooling as a safe but currently
  non-superior variant.
- The proposed latent repeated-visit availability/detection model is a fair and
  appropriate next step. The current two-stage bridge uses an aggregate
  availability point estimate as a fixed input to checklist-level detection,
  so zero-detection locality-seasons are already pushed toward low availability
  before detection uncertainty can feed back. The latent likelihood changes the
  estimand: zero-detection groups can mean either true non-availability or
  availability with missed detections across all visits.
- The first 20-epoch latent repeated-visit run is a working but underfit
  baseline. Group-level availability ranking is promising, but marginal
  checklist detection probabilities are strongly underpredicted. This is not a
  reason to abandon the path; it means the first latent model needs longer
  optimization and clearer diagnostics separating prior marginal detection,
  posterior availability, and conditional detection within known-positive
  locality-seasons.
- The 100-epoch latent repeated-visit run is a credible first latent baseline.
  It nearly closes the checklist-ranking gap to the two-component bridge, and
  the label-informed posterior diagnostic is stronger than the bridge, which
  indicates that repeated-visit availability information is useful. The fair
  prior marginal prediction still underpredicts detection and is less calibrated
  than the bridge, so the next work should focus on latent component scale and
  plausibility rather than another broad architecture change.
- The 200-epoch latent repeated-visit run produced only modest additional gains
  over e100. The prior marginal model moved slightly closer to the bridge, but
  underprediction remained. This suggests the remaining gap is not mainly
  optimizer runtime; it is a component-scale/identifiability issue in how
  availability and conditional detection trade off.
- The first global marginal-rate moment run (`marginal_rate_l2=100`, 100
  epochs) did exactly what the constraint was intended to do: it pulled prior
  marginal detection rates much closer to observed rates and improved pooled
  ECE/max-bin error. It also reduced ranking and worsened several species-level
  deltas, especially coastal/waterbird species. This makes it a useful
  sensitivity result but not a new preferred model yet.
- The 200-epoch marginal-rate sensitivity sweep is now confirming a smooth
  calibration/ranking frontier rather than a single dominant model. The
  `mrate25` run is currently the best diagnosed calibrated sensitivity: it
  improves BCE and calibration over unconstrained e200 with less pooled-ranking
  cost than `mrate100`. The `mrate50` headline metrics sit between `mrate25`
  and `mrate100`, and its saved-output diagnostics confirm that it is a viable
  midpoint rather than a clear replacement. Unconstrained e200 remains the
  ranking-oriented latent reference; `mrate25` and `mrate50` now bracket the
  useful calibrated sensitivity range.
- The first species-wise marginal-rate anchor (`mrate25_srate10`) is a useful
  but small nudge. It improves BCE, calibration, focus-season error, and
  availability AUPRC versus `mrate25`, with a tiny pooled AUPRC cost. It does
  not resolve the persistent species-level bridge losses, so the next useful
  test is one stronger species-wise anchor before closing this axis.
- The stronger species-wise anchor (`mrate25_srate50`) confirms that this axis
  is also a calibration/ranking tradeoff rather than a fix for the main species
  failures. It improves pooled calibration and focus-season error but reduces
  AUPRC and does not remove recurring losses for Red-breasted Nuthatch, Tree
  Swallow, Hooded Merganser, Laughing Gull, White-throated Sparrow, Bald Eagle,
  Mallard, Swamp Sparrow, and several other species. Close rate-penalty tuning
  for now and move to species/regime diagnostics.
- Saved-output species-pattern diagnostics confirm that the remaining latent
  losses are not a single-species-group problem. Mean AUPRC deltas versus the
  two-component bridge are negative for water/coastal, urban/generalist,
  open/agricultural, raptor/scavenger, and forest/woodland groups, while the
  "other" bucket is slightly positive. Persistent failures are therefore more
  likely tied to species-season detection/availability structure and component
  scale than to a simple waterbird/coastal artifact. The next modeling axis
  should be explicit species-season/phenology structure in the latent model,
  not another scalar marginal-rate penalty.
- Optional species-by-season offsets have now been added to the latent
  repeated-visit model. They default to off, so all earlier runs remain
  reproducible. The first full test should use a detection-side
  species-season offset with shrinkage, because the strongest diagnostic
  pattern is species-season prior-marginal underprediction rather than a
  single ecological-group failure.
- The first species-season detection-offset run
  (`latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p01`)
  is the first latent variant in this sequence that improves the balanced
  calibrated latent reference on ranking, BCE, and calibration at the same
  time. It improves prior-marginal micro AUPRC, BCE, ECE, max-bin error, and
  focus-season weighted absolute error relative to
  `latent_repeated_visit_e200_mrate25_srate10`. It still trails the
  two-component bridge on fair prior-marginal checklist AUPRC/BCE/calibration,
  and persistent species losses remain, so the next step should be a tight
  species-season shrinkage sensitivity rather than a new architecture.
- The looser species-season detection-offset run
  (`latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025`)
  is essentially tied with `0p01`, with tiny improvements in prior-marginal
  AUPRC, BCE, calibration, ECE, and focus-season weighted error. The difference
  is too small to justify more detection-side L2 tuning. Treat `0p0025` as the
  current best saved species-season detection sensitivity, but consider this
  axis closed unless a later diagnostic points back to it.
- Persistent latent losses after the species-season detection-offset sweep are
  still broad across groups, especially Tree Swallow, Hooded Merganser,
  Red-breasted Nuthatch, Mallard, Bald Eagle, White-throated Sparrow,
  Bufflehead, Swamp Sparrow, Double-crested Cormorant, Downy Woodpecker, and
  several nuthatch/woodland species. The next useful probe is availability-side
  species-season structure, because that tests whether the remaining gap is
  true seasonal availability/phenology rather than checklist-level seasonal
  detectability.
- The first species-season availability-offset run is not a replacement for the
  detection-side species-season model. It improves availability positive-triplet
  AUPRC and slightly improves pooled ECE/max-bin error, but it worsens fair
  prior-marginal checklist AUPRC, BCE, calibration error, and focus-season
  weighted error relative to the best detection-side run. It also does not
  remove the persistent species-loss pattern. Treat it as an informative
  sensitivity: seasonal availability structure matters, but availability-side
  offsets alone are not the missing piece.

Current tested-method timeline:

1. Bulk eBird preprocessing with complete-checklist detections, inferred
   modeled-species non-detections, covariate sampling, NC boundary/raster
   alignment, and stationary-distance handling.
2. Tabular baselines: prevalence, effort-only, ecology-only, combined, and MLP
   variants with spatial-stratified validation and calibration diagnostics.
3. Graph-ready species/checklist datasets and sampled-edge link baselines.
4. All-species checklist-batch bridge models, including stronger hybrid
   species-embedding models and RBF spatial residuals.
5. Spatial-cell GNNs with residual/separated-channel variants, access encoders,
   environmental-neighbor edges, species co-detection graphs, effort/access
   bias components, and support-aware gates.
6. Cross-regime validation across primary 10x10, coastal-stress, and
   regime-feature splits; the frozen-access residual GNN is modestly but
   consistently above matching tabular baselines, while gains remain small.
7. Locality-season replication dataset, replication diagnostics, and first
   aggregated binomial locality-season baseline.
8. First explicit two-component bridge script that uses an aggregate
   locality-season availability score as input to a checklist-level detection
   model with effort and timing features.
9. First two-component checklist detection run, showing clear ranking/BCE gains
   over one-component baselines and mostly sensible seasonal behavior.
10. County-season and focus-species season calibration diagnostics for the
    two-component bridge.
11. Filtered probability-bin diagnostics with a minimum-pair threshold so stable
    county-season and focus-species calibration failures can be separated from
    tiny-bin artifacts.
12. Focus-species monthly phenology and response-curve diagnostic outputs added
    and generated for the stable 10/20 two-component run.
13. Plausibility plotting and summary script added for monthly phenology,
    environmental-response curves, and two-component versus availability-only
    response-error differences.
14. First combined neutral-point shrinkage run completed. It improved
    calibration but produced mixed phenology/species effects, motivating
    one-penalty ablation rather than stronger combined shrinkage.
15. Residual-only shrinkage completed. It closely matched the combined result
    but was marginally worse on pooled ranking, monthly error, and max-bin
    calibration.
16. Availability-weight-only shrinkage completed. It was almost indistinguishable
    from the unregularized run, showing that residual shrinkage is the component
    responsible for both the broad calibration improvement and its tradeoffs.
17. Weaker residual-only shrinkage at `0.0025` completed. It split the difference
    between the unregularized and `0.01` runs and justified one final midpoint
    test at `0.005`.
18. Midpoint residual shrinkage at `0.005` completed. The full sequence showed
    a smooth tradeoff with no unique scalar optimum, so further global L2 tuning
    was stopped in favor of partial pooling.
19. Partially pooled effort responses completed at residual L2 `0.0025`. The
    result is almost indistinguishable from weak species-specific shrinkage, so
    the next ablation is a fully shared effort response rather than another
    scalar tuning pass.
20. Fully shared effort response completed at residual L2 `0.0025`. It degraded
    ranking sharply, confirming that species-specific effort/detection responses
    are required in this framework.
21. Added the first latent repeated-visit availability/detection model script.
    This is the first implementation of the repeated-visit likelihood rather
    than another checklist-level detector or residual architecture variant.
22. Ran the first full 20-epoch latent repeated-visit baseline. It produced
    promising group-level availability ranking but underpredicted marginal
    checklist detection rates, so the next step is a longer run with added
    latent-diagnostic outputs rather than changing the architecture.
23. Added latent diagnostic outputs that separate:
    - prior marginal checklist detection before knowing held-out group history
    - posterior marginal detection after conditioning on group detections
      (diagnostic only, because it is label-informed)
    - conditional detection within known-positive locality-seasons
      (diagnostic only, because these groups are confirmed available)
24. Set the next latent run to 100 epochs using the updated diagnostics. The
    purpose is to test whether the e20 marginal-detection underprediction is
    mainly optimization underfit before changing the latent architecture.
25. Ran the 100-epoch latent repeated-visit model. Checklist-level prior
    marginal prediction improved substantially and is now close to the
    two-component bridge on ranking, but it remains undercalibrated. The
    label-informed posterior diagnostic outperforms the bridge, supporting the
    repeated-visit availability direction.
26. Added `exp/diagnose_ebird_latent_repeated_visit.py` to compare latent runs
    against each other and against the two-component bridge without retraining.
    It writes headline comparisons, species deltas, availability species
    summaries, and focus-species season deltas.
27. Ran the 200-epoch latent repeated-visit model and diagnosed it against e100.
    It improved prior marginal AUPRC from 0.56294 to 0.56802 and BCE from
    0.30258 to 0.29878, but availability ranking slipped slightly and marginal
    underprediction persisted.
28. Added optional training-time marginal-rate moment penalties to
    `exp/ebird_locality_season_latent_model.py`:
    - `--marginal-rate-l2` anchors the overall prior marginal detection rate to
      the observed training detection rate
    - `--species-marginal-rate-l2` anchors species-wise prior marginal rates to
      observed training rates
    These are training constraints, not post-hoc calibration, and default to
    zero so all previous runs are reproducible.
29. Ran the first global marginal-rate moment probe:
    `latent_repeated_visit_e100_mrate100`. It improved pooled prior-marginal
    calibration but reduced ranking and worsened several species deltas,
    especially coastal/waterbird species.
30. Ran the apples-to-apples 200-epoch global marginal-rate probe:
    `latent_repeated_visit_e200_mrate100`. It retained the calibration gain
    with much smaller ranking and species-level cost than the e100
    moment-constrained run.
31. Ran and diagnosed the weaker 200-epoch `mrate25` probe. It improved BCE and
    prior-marginal calibration relative to unconstrained e200 while preserving
    more pooled ranking than `mrate100`, so it is now the best diagnosed
    calibrated latent sensitivity run.
32. Ran and diagnosed the 200-epoch `mrate50` probe. It is an intermediate
    calibration/ranking point: slightly better BCE/calibration than `mrate25`,
    slightly lower pooled AUPRC, and mixed species-level effects. This closes
    the current global marginal-rate sweep.
33. Added and ran `exp/compare_ebird_latent_repeated_visit_runs.py` for saved
    latent run comparisons. The mrate sweep summary confirms a smooth global
    frontier: stronger anchoring improves pooled/focus-season calibration but
    gradually lowers pooled AUPRC and can worsen some species losses.
34. Ran `latent_repeated_visit_e200_mrate25_srate10` and diagnosed it against
    `mrate25`. The weak species-wise marginal anchor slightly improved
    calibration, BCE, focus-season error, and availability AUPRC with only a
    tiny AUPRC cost, but it did not materially fix the main species-level losses.
35. Ran `latent_repeated_visit_e200_mrate25_srate50` and compared it across the
    full latent sensitivity set. It further improved pooled/focus-season
    calibration, but the cost to pooled AUPRC increased and the recurring
    species-level bridge losses persisted. This closes the current scalar
    rate-penalty tuning axis.
36. Added and ran `exp/diagnose_ebird_latent_species_patterns.py` on the full
    latent sensitivity set. The output shows broad, persistent losses across
    several species groups and identifies species-season underprediction as the
    next target before any additional latent architecture work.
37. Added optional species-by-season latent offsets to
    `exp/ebird_locality_season_latent_model.py`:
    `--species-season-mode {none,availability,detection,both}` and
    `--species-season-l2`. Smoke tests passed for the default path and the
    detection-offset path.
38. Ran the first full species-season detection-offset latent model with
    `--species-season-mode detection --species-season-l2 0.01`. This was a
    positive direction: it improved the balanced calibrated latent reference on
    AUPRC, BCE, and calibration, while leaving the two-component bridge ahead
    and several species-level losses unresolved.
39. Ran the looser species-season detection-offset sensitivity at
    `--species-season-l2 0.0025`. It was effectively tied with `0.01`, with
    very small pooled and focus-season improvements but no meaningful change to
    the persistent species-loss pattern. Detection-side species-season L2 tuning
    is therefore closed for now.
40. Ran the first species-season availability-offset sensitivity at
    `--species-season-mode availability --species-season-l2 0.01`. It improved
    availability ranking and slightly improved ECE/max-bin error, but it reduced
    prior-marginal checklist ranking, worsened BCE/calibration error, worsened
    focus-season weighted error, and did not resolve the persistent species
    losses. Availability-only species-season offsets are therefore not promoted.
41. Ran the combined species-season sensitivity with offsets in both
    availability and detection. It tied the best detection-only run on pooled
    AUPRC, modestly improved BCE and checklist calibration, slightly worsened
    focus-season error and availability ranking, and left the same persistent
    species losses. Parameter magnitudes show seasonal signal being split
    across both latent components. This closes the current species-season
    placement/L2 axis and shifts the next work to component and replication-
    support diagnostics.
42. Reproduced the preferred detection-only species-season run with compact
    component diagnostics. Fair group any-detection errors improve under
    stronger date and duration-bin support, while high-support checklist
    detection remains underpredicted. Known-positive conditional-detection
    diagnostics are close to aligned in the strongest support strata. Added
    optional stricter group-support filters so the next run can test whether
    stronger repeated-visit information stabilizes the latent separation.
43. Ran the stricter-support sensitivity requiring five dates and three
    duration bins. It improved several calibration summaries and produced
    nearly exact group any-detection calibration in the strongest support
    strata, but medium-support any-detection remained overpredicted and
    high-support checklist frequency remained underpredicted. This identifies
    conditional independence of repeated detections as the next assumption to
    test rather than supporting progressively stricter filtering.

How to maintain this document going forward:

- Keep raw command/result notes in the relevant timeline section.
- Add a short interpretation after each result, especially whether it changes
  the current preferred framework.
- Mark negative results explicitly rather than deleting them.
- Keep the top checkpoint current when the strategy changes.
- Keep the final implementation checklist focused on the active next steps,
  not every historical step already completed.

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
- **Locality-season**: A repeated-sampling unit formed by grouping checklists at
  the same eBird locality within a biological season/year window. It is the
  current bridge between checklist-level detection modeling and
  occupancy-style availability modeling.
- **Biological season-year**: A season label and year chosen to better match
  bird life-history periods than calendar months alone. In the current
  `biological-nc` scheme, winter spans December-February and December is
  assigned to the next winter year.
- **Availability**: The model's ecological/seasonal estimate that a species is
  plausibly available to be detected at a locality during a season. It is not
  identical to confirmed occupancy, but it is closer to the biological process
  than checklist-level detection.
- **Conditional detection probability (`p`)**: The modeled probability that a
  species is detected on a particular checklist given that it is available in
  that locality-season. Effort, protocol, observer count, time of day, and
  within-season timing can affect this component.
- **Prior marginal checklist detection (`psi * p`)**: The held-out probability
  of a checklist detection before using any detections from that held-out
  locality-season. This is the fair checklist-level prediction from the latent
  model.
- **Conditional any-detection probability**: For a species that is available in
  a locality-season with repeated checklists, the probability it is detected at
  least once: `1 - product(1 - p_i)`.
- **Prior group any-detection probability**: The observable group-level target
  `psi * (1 - product(1 - p_i))`. It can be compared fairly with whether the
  species was detected at least once in the locality-season. It is not the same
  as occupancy probability.
- **Known-positive-group detection diagnostic**: Conditional-detection
  evaluation restricted to locality-season/species groups with at least one
  observed detection. It is useful for interpreting `p`, but it is
  label-informed and selection-biased, so it is not a fair stand-alone test
  metric.
- **Replication support**: The amount and diversity of repeated observation
  information in a locality-season, including checklist count, distinct dates,
  effort bins, protocols, and observers. Stronger replication support should
  make availability/detection separation more stable.
- **Conditional independence of repeated detections**: The current latent
  likelihood assumes that checklist detections are independent after
  conditioning on species availability and each checklist's detection
  probability. Shared weather, abundance, observer skill, and local conditions
  can violate this assumption.
- **Pairwise co-detection probability**: The probability that a species is
  detected on two distinct checklists in the same locality-season. Under the
  independent model, its prior expectation is availability multiplied by the
  product of the two conditional detection probabilities. Under a frailty
  model, the product is integrated over the shared random effect. Comparing
  observed and predicted pairwise rates is a fair diagnostic of residual
  dependence.
- **Detection overdispersion / detection frailty**: Extra repeated-visit
  heterogeneity beyond the modeled checklist covariates. A shared latent
  detection propensity or random effect can induce positive correlation among
  visits without memorizing a specific locality, but it must be constrained so
  it does not absorb the availability component.
- **Hierarchical species frailty**: A shared global detection-frailty scale
  plus zero-centered species deviations. This preserves the broadly supported
  repeated-detection dependence while allowing regularized species departures,
  rather than estimating unrelated absolute scales for every species.
- **Adequately sampled locality-season**: A locality-season group with enough
  repeated and varied effort to be useful for availability/detection
  separation. The current flag requires at least two dates and at least two
  duration bins in addition to the minimum checklist count.
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

## Portable CONUS Covariate Pipeline

The next controlled modeling stage replaces the current coarse NC ecological
covariate block with a date-aware, multi-scale U.S. covariate system while
holding the promoted repeated-visit likelihood and transfer diagnostics fixed.
Architecture, source decisions, implementation status, exact commands, and NC
build results are maintained in
[ebird-conus-covariate-pipeline.md](ebird-conus-covariate-pipeline.md).

Phases 1-2 are complete: the checked-in registry contains 17 ecological, climate,
coastal, access, fallback, and evaluation source blocks, and the executable NC
plan resolves a 250 m `EPSG:5070` grid with 27 intersecting 100 km tiles. This
work is an input-data intervention, not another latent-model hyperparameter
search. Source blocks will be promoted by transfer and ecological-plausibility
evidence, with access variables kept out of availability unless an explicit
ablation justifies them. The generic COG/VRT engine has passed a synthetic
four-tile seam-equivalence test.

The Annual NLCD portion of Phase 3 is implemented against validated official
requester-pays rasters. The release-pinned NC catalog resolves 2019-2023 land
cover and 2020-2023 fractional imperviousness (nine archives, 10.2 GiB before
subsetting); 2019 is used only as the non-future predecessor for 2020 change.
The first full NC build produced 244 bands across 27 tiles (6,588 COGs,
2,108.33 MiB) in 1,993.3 seconds. Inventory, grid, range, class-fraction-sum,
and mapped coastal/interior QA passed.

Statewide checklist QA then exposed a raster-boundary contract issue that the
aggregate 99.97% support rate obscured. All 164 checklists lacking 250 m, 1 km,
and 5 km support are inside the vector NC AOI but within 98.11 m of its
boundary; their 250 m cell centers were outside and therefore masked. Six
additional 5 km-only failures are marine/pelagic and expected. The regional
grid contract is now `all_touched`, with vector point membership kept separate
and derived artifacts protected against reuse across mask rules. Annual NLCD
passes the revised 14-test regression suite. Replanning retains the same 27
tiles while adding only 5,295 boundary cells (`+0.2374%`; 2,235,542 total).
The full overwrite is now complete: 244 bands, 6,588 COGs, 2,112.22 MiB, and
2,227.7 seconds, with `all_touched` recorded in the summary. Corrected
numerical/mapped QA also passes: maximum class-fraction sum error is
`9.54e-07`, all automated checks pass, and seven representative previews show
coherent interior, state-edge, barrier-island, estuarine, and open-water
structure without visible seams. Post-rebuild checklist QA supports 661,978 of
661,979 events at 250 m and 1 km and 661,972 at 5 km. All-radius failures fell
from 164 to one; the remaining seven events are marine or marine-likely
traveling checklists plotted 3.49-6.66 km seaward. Annual NLCD is therefore
promoted with explicit terrestrial missingness for those events. The
repeated-visit likelihood, split, support rules, and optimizer remain frozen
while this ecological predictor
intervention is built. Full definitions and exact commands are maintained in
the dedicated covariate ledger.

The bounded LANDFIRE semantic and multi-release gate now passes. LF2016,
LF2022, and LF2023 each use a stable 46-band vegetation schema: nine portable
EVT class fractions and dominant-lifeform-conditional tree/shrub/herb cover
and height at 250 m, 1 km, and 5 km, plus modeled-source coverage. Interior and
coastal pilots pass COG, grid, value-range, VRT-order, sparse-band, mapped, and
EVT class-closure checks. LF2016/LF2022 maximum closure errors are at most
`8.34e-07`; LF2023 is `7.15e-07`. Coastal modeled support is consistently
99.04%, 97.81%, and 90.43% by increasing radius because neighborhoods reach
official LANDFIRE `Fill-NoData` ocean cells. This is retained as support
information, not imputed habitat.

Annual Dist20-Dist23 is also implemented as 12 year-and-radius-specific
disturbance-fraction bands. Official Background codes provide valid
undisturbed terrestrial support, while Fill and positive Water mask codes are
excluded from the denominator. Interior and coastal numerical and mapped QA
pass. The complete covariate regression suite now passes 42 tests. Vegetation
source age remains an observation-year/release provenance scalar for the
date-aware extractor rather than a spatially duplicated COG. Full-NC
LANDFIRE orchestration and named materialization profiles are next; the frozen
locality-season model is not yet refit.

```text
python scripts/data/ebird-covariates.py plan --config config/ebird_covariates/nc_2020_2023_v1.json

python scripts/data/ebird-covariates.py catalog-nlcd --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json

python scripts/data/ebird-covariates.py register-nlcd-aws --catalog data/ebird/covariates/raw/annual_nlcd/C1V2/catalog.json --output data/ebird/covariates/raw/annual_nlcd/C1V2/sources.aws.json

python scripts/data/ebird-covariates.py validate-nlcd-sources --sources data/ebird/covariates/raw/annual_nlcd/C1V2/sources.aws.json

python scripts/data/ebird-covariates.py derive-nlcd --plan data/ebird/covariates/builds/nc_2020_2023_covariates_v1/build_plan.json --sources data/ebird/covariates/raw/annual_nlcd/C1V2/sources.aws.json --output-dir data/ebird/covariates/builds/nc_2020_2023_covariates_v1/sources/annual_nlcd --overwrite
```

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

Historical note: this checkpoint records the state of the checklist-level GNN
work before the locality-season pivot. It remains useful for tracking tested
architectures and negative results, but the active modeling direction is now the
locality-season availability/detection framework described near the end of this
document and summarized at the top.

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

Implementation note:

- `exp/ebird_joint_tabular_baseline.py` now supports
  `--feature-set both-regime`.
- `both-regime` keeps the current `both` features and adds:
  - `is_coastal_25km`
  - `is_near_water_2p5km`
  - `coastal_x_traveling`
  - `coastal_x_duration_log1p`
  - `coastal_x_effort_distance_log1p`
  - `near_water_x_traveling`
  - `near_water_x_duration_log1p`
- The waterbody and coastline distance columns were already log-transformed in
  the modeling feature builder despite retaining their original column names.
  The new feature set therefore tests regime indicators and effort interactions,
  not new source data.
- `exp/build_ebird_graph_dataset.py` also supports `--feature-set both-regime`,
  so we can build a separate graph dataset without changing the existing
  `both` graph outputs.
- Spatial GNN component/channel inference treats coastal and near-water
  indicators as ecological cell features and the coastal/near-water effort
  interactions as access/bias features.

First `both-regime` tabular comparison commands:

```
python exp/ebird_joint_tabular_baseline.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --feature-set both-regime --model mlp --split spatial-stratified --spatial-blocks-per-dim 10 --test-fraction 0.2 --epochs 50 --hidden-dim 64 --hidden-layers 1 --dropout 0.10
python exp/compare_ebird_tabular_baselines.py --top-species 100 --split spatial-stratified --model mlp
python exp/compare_ebird_calibration.py --top-species 100 --split spatial-stratified --model mlp
```

`both-regime` tabular result:

- Same 10x10 spatial-stratified split as the current graph sandbox:
  - held-out blocks 65, 79, and 31
  - 137,020 test checklists
  - 524,959 train checklists
- Current `both` MLP:
  - macro AUROC 0.8404
  - macro AUPRC 0.4017
  - micro AUROC 0.8897
  - micro AUPRC 0.5725
  - ECE 0.0051
  - max bin error 0.0201
- `both-regime` MLP:
  - macro AUROC 0.8416
  - macro AUPRC 0.4056
  - micro AUROC 0.8897
  - micro AUPRC 0.5739
  - ECE 0.0062
  - max bin error 0.0207
- Interpretation:
  - `both-regime` gives a small aggregate ranking lift: +0.0039 macro AUPRC
    and +0.0014 micro AUPRC.
  - Calibration is slightly worse: ECE increases from 0.0051 to 0.0062.
  - The strongest species gains are concentrated in coastal/water-associated
    species: Double-crested Cormorant (+0.0917 AUPRC), Great Egret (+0.0770),
    Ring-billed Gull (+0.0562), Killdeer (+0.0552), Bald Eagle (+0.0536),
    Osprey (+0.0399), Royal Tern (+0.0354), Boat-tailed Grackle (+0.0337),
    Pied-billed Grebe (+0.0253), Bufflehead (+0.0225), and Brown Pelican
    (+0.0197).
  - The largest losses include House Sparrow (-0.0435), Eastern Meadowlark
    (-0.0300), Hooded Warbler (-0.0263), Green Heron (-0.0184), Field Sparrow
    (-0.0184), Northern Mockingbird (-0.0178), and Blue Grosbeak (-0.0178).
- Comparison-script note:
  - The comparison scripts now skip incompatible split configurations. This was
    needed because old 8x8 and new 10x10 spatial-stratified outputs share the
    same historical filename pattern.
  - In this run, effort and ecology MLP outputs were skipped because they did
    not match the current 10x10 split signature.

Decision:

- Carry `both-regime` into one clean graph/GNN run as a targeted coastal-regime
  test, because it improves exactly the species group that motivated the
  feature change.
- Do not treat it as the new default yet because the aggregate gain is small and
  calibration is slightly worse.

If `both-regime` improves the tabular MLP, build a regime-feature graph dataset:

```
python exp/build_ebird_graph_dataset.py --processed-dir data/ebird/processed_nc_2020_2023 --output-dir data/ebird/graph_top100_spatial_10x10_regime --top-species 100 --feature-set both-regime --split spatial-stratified --spatial-blocks-per-dim 10 --test-fraction 0.2 --negative-ratio 5 --overwrite
python exp/validate_ebird_graph_dataset.py --graph-dir data/ebird/graph_top100_spatial_10x10_regime
```

Then train the clean frozen-access spatial GNN equivalent on the regime graph:

```
python exp/train_ebird_access_encoder.py --graph-dir data/ebird/graph_top100_spatial_10x10_regime --run-name access_gcn_h64_l2_z64 --epochs 500 --hidden-dim 64 --layers 2 --embedding-dim 64 --dropout 0.10 --spatial-grid-size-m 25000
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10_regime --run-name spatial_gcn_frozen_access_h64_l2_z64 --gnn-mode residual --component-mode joint --spatial-channel-mode separated --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-access-bias-l2 0.001 --frozen-access-embeddings data/ebird/graph_top100_spatial_10x10_regime/access_encoder/access_gcn_h64_l2_z64_cell_embeddings.npy
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10_regime --spatial-run-name spatial_gcn_frozen_access_h64_l2_z64
python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10_regime --spatial-run-name spatial_gcn_frozen_access_h64_l2_z64
```

`both-regime` graph build and access encoder:

- `data/ebird/graph_top100_spatial_10x10_regime` was built and validated.
- Counts match the current 10x10 graph sandbox:
  - 661,979 checklists
  - 100 species
  - 524,959 train checklists
  - 137,020 test checklists
  - 9,017,501 positive edges
  - 35,150,161 sampled negative edges
  - 7,149,369 train positive edges
  - 1,868,132 test positive edges
  - 27,797,825 train negative edges
  - 7,352,336 test negative edges
- Feature matrix is 661,979 x 21, as expected for the seven added regime
  features.
- Access encoder `access_gcn_h64_l2_z64` completed 500 epochs.
- Validation MSE reached its best value around epoch 150 and ended at 0.5936.
- Target-wise validation performance:
  - number observers: Pearson 0.5563
  - duration: Pearson 0.2911
  - effort distance: Pearson 0.3275
  - locality per checklist: Pearson 0.4731
  - observer per checklist: Pearson 0.4325
  - stationary/traveling rates: Pearson about 0.14
  - log train checklists: Pearson 0.4794
  - log unique localities: Pearson 0.4056
- Compared with the previous non-regime access encoder, this is similar but
  slightly weaker for several access targets. Treat it as adequate for the
  targeted regime-feature GNN test, not evidence that the regime features
  improved the access encoder itself.

Command used for the targeted regime GNN test:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10_regime --run-name spatial_gcn_frozen_access_h64_l2_z64 --gnn-mode residual --component-mode joint --spatial-channel-mode separated --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-access-bias-l2 0.001 --frozen-access-embeddings data/ebird/graph_top100_spatial_10x10_regime/access_encoder/access_gcn_h64_l2_z64_cell_embeddings.npy
```

`both-regime` frozen-access spatial GNN result:

- `spatial_gcn_frozen_access_h64_l2_z64` on
  `data/ebird/graph_top100_spatial_10x10_regime`
- Training BCE:
  - epoch 1: 0.31310
  - epoch 5: 0.26089
  - epoch 10: 0.25507
- Test metrics:
  - micro AUROC 0.8906
  - micro AUPRC 0.5762
  - macro AUROC 0.8441
  - macro AUPRC 0.4118
  - ECE 0.0114
  - max bin error 0.0409
  - species calibration MAE 0.0165

Interpretation:

- This is worse than the clean non-regime frozen-access GNN on the current
  10x10 split:
  - previous clean GNN: micro AUPRC about 0.5800, macro AUPRC about 0.4151,
    ECE about 0.0056, species calibration MAE about 0.0133
  - regime GNN: micro AUPRC 0.5762, macro AUPRC 0.4118, ECE 0.0114, species
    calibration MAE 0.0165
- The tabular MLP gained from explicit regime features, especially for
  coastal/water-associated species, but the frozen-access spatial GNN did not
  benefit in aggregate and calibration worsened materially.
- Do not promote `both-regime` to the default GNN feature set based on this run.

`both-regime` frozen-access GNN diagnostics:

- Effort-strata comparison against the regime tabular MLP:
  - block 65 gains: +0.0085 micro AUPRC, +0.0103 macro AUPRC, but ECE worsens
    by 0.0056.
  - longer or higher-distance effort still benefits modestly, e.g. duration
    121+ (+0.0055 micro AUPRC), distance `(2,5]` (+0.0065), and traveling
    (+0.0033).
  - block 79 loses badly: -0.0308 micro AUPRC, -0.0091 macro AUPRC, and ECE
    worsens by 0.0092.
  - block 31 is mixed: -0.0030 micro AUPRC but +0.0034 macro AUPRC.
- Block/species diagnostics:
  - block 79 has mean delta AUPRC -0.0096, median -0.0051, 59 losses and 38
    gains.
  - block 65 has mean delta AUPRC +0.0105, median +0.0057, 26 losses and 69
    gains.
  - block 31 has mean delta AUPRC +0.0035, median +0.0008, 43 losses and 52
    gains.
  - block 79 gains include Northern Parula, Turkey Vulture,
    Double-crested Cormorant, American Herring Gull, and Great Black-backed
    Gull.
  - block 79 losses include House Sparrow, Great Egret, Red-bellied Woodpecker,
    European Starling, Purple Martin, Downy Woodpecker, Eastern Phoebe,
    Carolina Chickadee, Northern Cardinal, and Red-winged Blackbird.

Decision:

- Keep `both-regime` as a useful tabular diagnostic feature set because it
  exposes coastal/water regime effects and helped several coastal species in the
  MLP baseline.
- Do not carry `both-regime` forward as the main GNN feature set for the current
  NC top-100 sandbox. The regime frozen-access GNN worsens aggregate ranking,
  calibration, and the already problematic coastal held-out block.
- Treat the block 79 issue as a validation-design and transfer-support issue,
  not a reason to keep adding handcrafted coastal interaction features. The
  current split has only one fully coastal held-out block, and that block has a
  distinct access/observer regime.

Diagnostic-output note:

- `compare_ebird_effort_strata.py` and `diagnose_ebird_block_species.py` now
  write run-specific copies under a subfolder named after `--spatial-run-name`
  while also refreshing the previous compatibility files. This avoids confusing
  a clean frozen-access diagnostic with a later species-GCN or regime diagnostic
  that reused the same generic output directory.

Recommended next step:

- Return to the clean non-regime frozen-access GNN as the primary branch.
- Re-run clean non-regime diagnostics once with the updated run-specific output
  behavior so future comparisons have unambiguous files.
- Then improve validation geometry before adding more model variants: create
  either a multi-fold spatial blocked evaluation or a split that explicitly
  balances coastal/near-water/inland blocks, effort/access strata, and common
  species prevalence.

Clean diagnostic refresh commands:

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_frozen_access_h64_l2_z64
python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --spatial-run-name spatial_gcn_frozen_access_h64_l2_z64
```

Clean non-regime frozen-access diagnostic refresh:

- Diagnostics now wrote run-specific outputs to:
  - `data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/diagnostics/effort_strata/spatial_gcn_frozen_access_h64_l2_z64`
  - `data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/diagnostics/block_species/spatial_gcn_frozen_access_h64_l2_z64`
- Effort-strata result:
  - The clean frozen-access GNN improves over the tabular MLP in most effort
    strata.
  - Largest micro-AUPRC gains are for long/high-effort checklists: duration
    121+ (+0.0134), distance `(2,5]` (+0.0130), 3+ observers (+0.0116),
    distance 5+ (+0.0106), traveling (+0.0095).
  - Spatial block 65 improves by +0.0101 micro AUPRC and +0.0101 macro AUPRC.
  - Spatial block 31 has a smaller positive result: +0.0036 micro AUPRC and
    +0.0024 macro AUPRC.
  - Spatial block 79 remains the only block-level loss: -0.0057 micro AUPRC and
    -0.0070 macro AUPRC. Its ECE is slightly better than tabular (-0.0023), so
    the issue is primarily ranking/transfer, not just probability calibration.
- Block/species result:
  - block 65: mean delta AUPRC +0.0102, median +0.0059, 67 species gains and
    28 losses.
  - block 31: mean delta AUPRC +0.0025, median +0.0018, 59 species gains and
    36 losses.
  - block 79: mean delta AUPRC -0.0079, median -0.0010, 44 species gains and
    53 losses.
  - block 79 still has meaningful coastal/water species gains, including
    Ring-billed Gull (+0.0655), Yellow-rumped Warbler (+0.0604), Canada Goose
    (+0.0521), Mallard (+0.0484), Great Black-backed Gull (+0.0474), and
    Bufflehead (+0.0468).
  - The largest block 79 losses are mostly common/developed-area or generalist
    species: House Finch (-0.2491), House Sparrow (-0.2183), European Starling
    (-0.1115), Mourning Dove (-0.0797), Northern Mockingbird (-0.0733),
    Northern Cardinal (-0.0728), Downy Woodpecker (-0.0597), and Carolina Wren
    (-0.0495).

Current interpretation:

- The clean frozen-access GNN is still the best current branch. It is not
  universally better, but its failures are geographically concentrated rather
  than spread across all effort strata.
- The repeated issue is not that temporal/environmental/effort covariates are
  missing; they are in the model. The weakness is that the current single
  spatial split asks the model to transfer into one distinctive coastal/access
  regime represented by block 79.
- The next framework-level step should be better validation geometry, not more
  hand-tuned coastal features. We need to know whether the GNN fails generally
  on coastal transfer or whether this specific held-out block is unusually hard.

Split-candidate diagnostic:

- `exp/diagnose_ebird_split_candidates.py` scores candidate spatial split seeds
  and grid sizes before rebuilding graph datasets.
- It uses the same existing spatial-stratified split machinery, then reports
  balance error, species-prevalence balance, coastal/near-water test coverage,
  effort/access balance, and selected block IDs.
- The score is only a screening heuristic. The goal is to find candidate splits
  with more representative coastal/near-water holdouts before doing expensive
  graph builds and model runs.

Next command:

```
python exp/diagnose_ebird_split_candidates.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --spatial-blocks-per-dim 10 --seeds 1-100 --test-fraction 0.2 --stratify-species-count 20 --output data/ebird/split_diagnostics/top100_10x10_split_candidates.csv
```

Initial split-candidate result:

- The seed sweep selected the same 10x10 held-out blocks for every seed:
  blocks 65, 79, and 31.
- This means the current greedy selector has a stable best solution under its
  present objective; the random seed is not giving meaningfully different
  spatial validation scenarios.
- The selected test set has:
  - 20.7% of checklists
  - balance error 0.0525
  - species-prevalence MAE 0.0238
  - one coastal held-out block and one near-water held-out block
  - test coastal rate 0.1712 vs train coastal rate 0.2150
- Interpretation: the current split is not obviously bad by aggregate balance,
  but it does not give enough independent coastal validation blocks to diagnose
  whether block 79 is a one-off failure or a general coastal-transfer weakness.

The split diagnostic now supports `--mode exhaustive`, which searches candidate
block combinations directly instead of only replaying the greedy seed-dependent
selector. The first exhaustive version was too slow because it rescanned all
checklists for many candidate block sets. The script now scores exhaustive
candidates from precomputed block-level summaries and includes a hard
`--max-combinations` cap.

Next split-search command:

```
python exp/diagnose_ebird_split_candidates.py --processed-dir data/ebird/processed_nc_2020_2023 --top-species 100 --spatial-blocks-per-dim 10 --mode exhaustive --min-test-blocks 3 --max-test-blocks 5 --max-combinations 500000 --max-candidates 50000 --size-tolerance 0.08 --test-fraction 0.2 --stratify-species-count 20 --output data/ebird/split_diagnostics/top100_10x10_exhaustive_split_candidates.csv
```

Exhaustive split-search result:

- The best coastal-coverage alternatives all include block 79 plus additional
  coastal or near-coastal blocks. The top candidate is blocks 17, 48, 65, and
  79.
- Top candidate:
  - test fraction 0.1985
  - balance error 0.0992
  - species-prevalence MAE 0.0368
  - three coastal test blocks
  - two near-water test blocks
  - test coastal rate 0.4023 vs train coastal rate 0.1573
- This is not a better balanced replacement for the primary split. It is a
  coastal-transfer stress split. It intentionally holds out much more coastal
  effort than the training set, so aggregate metrics will likely drop.
- Use this to ask a specific framework question: does the clean frozen-access
  GNN fail generally when transferring into coastal/near-water regimes, or was
  the original block 79 result partly a single-block artifact?

Implementation note:

- `exp/ebird_joint_tabular_baseline.py` and `exp/build_ebird_graph_dataset.py`
  now support `--test-block-ids` for fixed spatial-block holdouts. This keeps
  the split definition explicit and reproducible for stress tests.

Recommended coastal stress-test graph:

```
python exp/build_ebird_graph_dataset.py --processed-dir data/ebird/processed_nc_2020_2023 --output-dir data/ebird/graph_top100_spatial_10x10_coastalstress --top-species 100 --feature-set both --split spatial-stratified --spatial-blocks-per-dim 10 --test-block-ids "17 48 65 79" --negative-ratio 5 --overwrite
python exp/validate_ebird_graph_dataset.py --graph-dir data/ebird/graph_top100_spatial_10x10_coastalstress
```

Coastal stress graph build result:

- `data/ebird/graph_top100_spatial_10x10_coastalstress` was built and
  validated.
- Fixed held-out blocks: 17, 48, 65, and 79.
- Feature set: `both`.
- Feature matrix: 661,979 x 14.
- Counts:
  - 661,979 checklists
  - 100 species
  - 530,558 train checklists
  - 131,421 test checklists
  - 9,017,501 positive edges
  - 35,150,161 sampled negative edges
  - 7,183,089 train positive edges
  - 1,834,412 test positive edges
  - 28,047,863 train negative edges
  - 7,102,298 test negative edges
- Validation passed. This graph is ready for matched access-encoder and clean
  frozen-access GNN training.

Train the same clean frozen-access branch on the coastal stress split:

```
python exp/train_ebird_access_encoder.py --graph-dir data/ebird/graph_top100_spatial_10x10_coastalstress --run-name access_gcn_h64_l2_z64 --epochs 500 --hidden-dim 64 --layers 2 --embedding-dim 64 --dropout 0.10 --spatial-grid-size-m 25000
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10_coastalstress --run-name spatial_gcn_frozen_access_h64_l2_z64 --gnn-mode residual --component-mode joint --spatial-channel-mode separated --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-access-bias-l2 0.001 --frozen-access-embeddings data/ebird/graph_top100_spatial_10x10_coastalstress/access_encoder/access_gcn_h64_l2_z64_cell_embeddings.npy
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10_coastalstress --spatial-run-name spatial_gcn_frozen_access_h64_l2_z64
python exp/diagnose_ebird_block_species.py --graph-dir data/ebird/graph_top100_spatial_10x10_coastalstress --spatial-run-name spatial_gcn_frozen_access_h64_l2_z64
```

Coastal stress frozen-access result:

- Access encoder:
  - final train MSE: 0.3112
  - final validation MSE: 0.5317
  - strongest validation correlations were observer/checklist density,
    duration, and checklist density; stationary/traveling rates remained weak.
- Spatial-cell GNN:
  - micro AUROC: 0.8868
  - micro AUPRC: 0.5767
  - macro AUROC: 0.8399
  - macro AUPRC: 0.4092
  - ECE: 0.0012
  - max bin error: 0.0086
  - species calibration MAE: 0.0109
- Interpretation:
  - Ranking metrics are lower than the primary 10x10 frozen-access run, which
    is expected because the fixed held-out blocks deliberately stress coastal
    transfer.
  - Calibration is very strong, so the main question is not whether predicted
    probabilities are globally sane. The next diagnostic question is whether
    rank losses concentrate in coastal blocks/species, or whether the stress
    split exposes broader transfer weaknesses.

Coastal stress diagnostics:

- Effort-strata comparison:
  - Most effort strata still improved over the tabular baseline.
  - Largest micro-AUPRC gains were in block 17 (+0.0227), 3+ observers
    (+0.0171), long-distance checklists (+0.0159), 2 observers (+0.0132),
    2-5 km traveling effort (+0.0130), and long-duration checklists.
  - Block 79 improved in aggregate under this stress split (+0.0097
    micro-AUPRC, +0.0077 macro-AUPRC), unlike the primary 10x10 split where it
    was the main weak block.
  - Block 65 remained positive (+0.0093 micro-AUPRC).
  - Block 48 was the only block-level loss (-0.0078 micro-AUPRC,
    -0.0028 macro-AUPRC).
- Block/species comparison:
  - Mean block AUPRC deltas:
    - block 17: +0.0199, 73 species gained and 27 lost
    - block 79: +0.0079, 59 species gained and 38 lost
    - block 65: +0.0063, 62 species gained and 33 lost
    - block 48: -0.0028, 45 species gained and 54 lost
  - The largest gains included Pied-billed Grebe in block 17, Brown-headed
    Nuthatch in block 79, Black-and-white Warbler in block 65, Green Heron in
    block 48, and several water/edge-associated species in block 17.
  - The largest losses were species/block specific, especially House Finch in
    block 79 (-0.2393 AUPRC), Red-headed Woodpecker in block 65 (-0.1019),
    House Sparrow in block 79 (-0.0963), and several block-48 species.
- Interpretation:
  - The original block-79 weakness does not generalize cleanly to every
    coastal-stress split. With additional coastal blocks held out, block 79
    improves in aggregate.
  - This points away from a simple "coastal species fail" diagnosis and toward
    split-specific transfer behavior: some species/block combinations remain
    brittle, but the framework is not uniformly worse in coastal regimes.
  - Next step: diagnose block 48 directly and compare the coastal-stress split
    against the primary split at the species level. The goal is to distinguish
    framework limitations from single-split artifacts before changing model
    architecture again.
- Regime/support diagnostic:
  - Held-out blocks 48, 79, and 17 are all coastal-dominated:
    - block 48: coastal rate 0.9999, near-water rate 0.4053
    - block 79: coastal rate 1.0000, near-water rate 0.7611
    - block 17: coastal rate 0.9734, near-water rate 0.6423
  - Block 65 is inland and low near-water:
    - block 65: coastal rate 0.0000, near-water rate 0.0331
  - Nearest-train ecological distances for coastal held-out blocks were similar
    enough that block 48 does not look uniquely unsupported on ecology alone:
    - block 48: 0.4524
    - block 79: 0.4426
    - block 17: 0.4268
    - block 65: 0.3425
  - Block 79 had much higher nearest-train access distance (1.0318) than block
    48 (0.4421) or block 17 (0.4715), but block 79 still improved in aggregate.
    This reinforces that access support alone is not explaining the remaining
    block/species failures.
- Residual-map summary:
  - For the mapped focus species, access-channel probability deltas were small
    relative to the full spatial residual deltas.
  - Several species were shifted downward overall, including Red-headed
    Woodpecker, House Sparrow, American Redstart, Green Heron, Pied-billed
    Grebe, Belted Kingfisher, and Mallard.
  - House Finch and Brown-headed Nuthatch had positive full probability deltas
    on average, even though House Finch remains a major block-79 ranking loss.
  - This suggests the remaining failures are mostly species-specific spatial
    ranking issues rather than simple global over/under-calibration or a
    dominant access-bias error.
- Matched coastal-stress tabular comparison:
  - A matching MLP tabular baseline was run with the same fixed held-out blocks
    `17 48 65 79`, so the graph-vs-tabular comparison is now on the same
    split.
  - Matched tabular MLP:
    - macro AUROC: 0.8346
    - macro AUPRC: 0.3966
    - micro AUROC: 0.8834
    - micro AUPRC: 0.5676
    - ECE: 0.0021
  - Frozen-access spatial GNN on the same split:
    - macro AUROC: 0.8399
    - macro AUPRC: 0.4092
    - micro AUROC: 0.8868
    - micro AUPRC: 0.5767
    - ECE: 0.0012
  - Interpretation:
    - The spatial GNN is only modestly better in aggregate, but the gain is
      consistent across AUROC, AUPRC, and calibration.
    - Because the comparison uses all held-out checklist/species pairs for both
      models, the AUPRC differences are comparable.
  - Largest species-level graph gains over tabular included:
    - Double-crested Cormorant: +0.1225 AUPRC
    - Black-and-white Warbler: +0.0987 AUPRC
    - Ring-billed Gull: +0.0606 AUPRC
    - Bald Eagle: +0.0568 AUPRC
    - Yellow-throated Warbler: +0.0546 AUPRC
    - Belted Kingfisher: +0.0452 AUPRC
    - Osprey: +0.0437 AUPRC
  - Largest species-level graph losses versus tabular included:
    - Red-headed Woodpecker: -0.0621 AUPRC
    - European Starling: -0.0222 AUPRC
    - Green Heron: -0.0197 AUPRC
    - Northern Rough-winged Swallow: -0.0180 AUPRC
    - Dark-eyed Junco: -0.0089 AUPRC
  - Current reading:
    - The GNN is adding useful spatial/species structure, especially for some
      coastal, aquatic, and warbler species.
    - The persistent losses are a smaller set of species where the spatial
      residual appears to hurt ranking despite generally good calibration.
    - This argues for species-level stability diagnostics and residual
      regularization/inspection before adding more architecture.
- Species-stability diagnostic:
  - Use `exp/compare_ebird_species_stability.py` to compare graph-minus-tabular
    species deltas across the primary 10x10 split and the coastal-stress split.
  - This labels species as consistently helped, consistently hurt, or
    split-sensitive, using a default absolute delta threshold of 0.005 AUPRC.
  - Command:

```
python exp/compare_ebird_species_stability.py --primary data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/residual_primary_graph_vs_tabular_species.csv --comparison data/ebird/graph_top100_spatial_10x10_coastalstress/spatial_gnn_baselines/spatial_gcn_frozen_access_h64_l2_z64_graph_vs_tabular_species.csv --primary-label primary --comparison-label coastalstress --metric auprc --threshold 0.005 --output data/ebird/graph_top100_spatial_10x10_coastalstress/spatial_gnn_baselines/diagnostics/species_stability/primary_vs_coastalstress_species_stability.csv
```

Species-stability result:

- Joined 100 species across the primary and coastal-stress graph-vs-tabular
  comparisons.
- Mean graph-minus-tabular AUPRC:
  - primary split: -0.0093
  - coastal-stress split: +0.0125
- Median graph-minus-tabular AUPRC:
  - primary split: -0.0096
  - coastal-stress split: +0.0096
- Stability classes:
  - consistently helped: 26 species
  - consistently hurt: 8 species
  - primary hurt, coastal-stress helped: 29 species
  - primary helped, coastal-stress hurt: 3 species
  - primary-only helped: 6 species
  - primary-only hurt: 22 species
  - comparison-only helped: 4 species
  - neutral/small: 2 species
- Consistently helped examples:
  - Black-and-white Warbler: +0.0981 primary, +0.0987 coastal-stress
  - Northern Mockingbird: +0.0856 primary, +0.0341 coastal-stress
  - Hooded Warbler: +0.0311 primary, +0.0313 coastal-stress
  - American Robin: +0.0711 primary, +0.0235 coastal-stress
  - Eastern Towhee: +0.0776 primary, +0.0230 coastal-stress
  - Great Egret, Indigo Bunting, Pine Warbler, Brown-headed Nuthatch, Blue
    Grosbeak, Osprey, and Boat-tailed Grackle were also consistently helped.
- Consistently hurt examples:
  - Red-headed Woodpecker: -0.1489 primary, -0.0621 coastal-stress
  - Northern Rough-winged Swallow: -0.0387 primary, -0.0180 coastal-stress
  - Green Heron: -0.0116 primary, -0.0197 coastal-stress
  - Dark-eyed Junco: -0.0108 primary, -0.0089 coastal-stress
  - Scarlet Tanager, Swamp Sparrow, Hairy Woodpecker, and Common Grackle were
    also consistently hurt.
- Split-sensitive species:
  - Several coastal/water species flipped from primary losses to coastal-stress
    gains: Double-crested Cormorant, Bufflehead, Pied-billed Grebe,
    Ring-billed Gull, Brown Pelican, Hooded Merganser, Belted Kingfisher, Bald
    Eagle, and Great Blue Heron.
  - European Starling flipped the other way: primary helped, coastal-stress
    hurt.
- Current interpretation:
  - The framework is not merely failing on coastal or water species; many of
    those species are helped under the coastal-stress split.
  - The primary split was likely too dependent on one difficult coastal holdout
    and understated graph value for several coastal/water species.
  - The most useful next model-development target is the consistently hurt
    group, especially Red-headed Woodpecker. Those species may indicate where
    the spatial residual over-regularizes, borrows misleading neighbor signal,
    or fails to preserve local ecological specificity.
  - Before adding more architecture, use the stability classes to drive focused
    residual maps and species-specific diagnostics.
- Consistently hurt species residual maps:
  - Focus species: Red-headed Woodpecker, Northern Rough-winged Swallow, Green
    Heron, Dark-eyed Junco, Scarlet Tanager, Swamp Sparrow, Hairy Woodpecker,
    and Common Grackle.
  - Access-channel deltas were small for all focus species, so these losses do
    not look like access-bias domination.
  - Several consistently hurt species had strong negative full spatial deltas:
    - Red-headed Woodpecker: mean delta -0.0331, positive mean delta -0.0697
    - Green Heron: mean delta -0.0216, positive mean delta -0.0638
    - Scarlet Tanager: mean delta -0.0263, positive mean delta -0.1326
    - Hairy Woodpecker: mean delta -0.0384, positive mean delta -0.0750
  - Northern Rough-winged Swallow also shifted downward, but less strongly:
    mean delta -0.0109, positive mean delta -0.0211.
  - Dark-eyed Junco and Common Grackle were different:
    - Dark-eyed Junco had a negative mean delta overall but a positive mean
      delta on positives, suggesting ranking degradation may come from the
      relative treatment of negatives rather than blanket suppression.
    - Common Grackle had a positive mean full delta and positive mean delta on
      positives, suggesting its AUPRC loss is likely a ranking/spread issue
      rather than underprediction.
  - Current interpretation:
    - The main recurring failure mode is not the access component. It is the
      species-specific spatial residual suppressing positives too much for some
      species, especially forest/edge or locally structured species.
    - The next diagnostic should separate residual behavior by block for these
      species, because the all-heldout summary can hide whether one block is
      driving the species-level loss.
- Block-specific residual diagnostic:
  - `exp/diagnose_ebird_species_block_residuals.py` summarizes the GNN's own
    residual correction by held-out block and species.
  - It reports base and full probabilities, full-minus-base deltas, access
    deltas when available, AUROC/AUPRC changes, and calibration-error changes.
  - Note: the first implementation recomputed block IDs using only the test
    subset extent, which produced invalid block IDs for the coastal-stress
    split. The script now assigns blocks from the full graph extent, matching
    the split definition. Discard earlier output with block IDs outside
    `17, 48, 65, 79` for this graph.
  - Command:

```
python exp/diagnose_ebird_species_block_residuals.py --graph-dir data/ebird/graph_top100_spatial_10x10_coastalstress --run-name spatial_gcn_frozen_access_h64_l2_z64 --species "Red-headed Woodpecker" "Northern Rough-winged Swallow" "Green Heron" "Dark-eyed Junco" "Scarlet Tanager" "Swamp Sparrow" "Hairy Woodpecker" "Common Grackle"
```
  - Corrected result:
    - The valid held-out blocks were `17`, `48`, `65`, and `79`.
    - The largest full-vs-base AUPRC loss was Red-headed Woodpecker in block
      65: -0.0490 AUPRC, with positive mean delta -0.0449.
    - Green Heron losses were concentrated in blocks 17 and 65:
      - block 17: -0.0211 AUPRC, positive mean delta -0.0932
      - block 65: -0.0108 AUPRC, positive mean delta -0.0471
    - Red-headed Woodpecker also showed strong positive suppression in blocks
      17, 48, and 79, but those blocks did not all translate into AUPRC losses:
      - block 17: positive mean delta -0.1504, AUPRC +0.0123
      - block 48: positive mean delta -0.1021, AUPRC -0.0131
      - block 79: positive mean delta -0.1412, AUPRC +0.0334
    - Scarlet Tanager had severe positive suppression in blocks 17 and 65, but
      block 65 still gained AUPRC:
      - block 17: positive mean delta -0.1178, AUPRC -0.0072
      - block 65: positive mean delta -0.1338, AUPRC +0.0118
    - Hairy Woodpecker losses were small and concentrated in blocks 65, 79, and
      48.
    - Dark-eyed Junco, Northern Rough-winged Swallow, Green Heron, and Common
      Grackle all show cases where calibration improves or mean probabilities
      move in a plausible direction but ranking can still worsen.
  - Current interpretation:
    - Positive suppression alone is not sufficient to explain AUPRC loss. Some
      blocks show strong downward residual shifts on positives but improved
      AUPRC, meaning the residual may also be suppressing negatives enough to
      improve ranking.
    - The more actionable failure mode is block/species ranking distortion:
      Red-headed Woodpecker in block 65 and Green Heron in blocks 17/65 are the
      clearest examples where the residual hurts ranking and suppresses
      positives.
    - This argues for residual regularization or gating that is species- and
      support-aware, rather than simply shrinking all spatial residuals.
    - A useful next model change would be to penalize residuals more strongly
      where local species support is weak, or to add a species-specific residual
      gate that can learn to trust the base tabular path for sensitive species.

## Training Objective Options

## Support-Aware Residual Gate

Current model-development direction:

- The most generalizable framework remains a portable base detector plus a
  constrained spatial/species residual GNN:
  `base ecological/temporal/effort model + support gate * spatial residual`.
- This is preferable to species-by-species tuning because the decision to trust
  or dampen the residual is based on general support information, not on
  hand-selected species or NC-specific failure cases.
- The first implementation adds `--support-aware-residual cell` to
  `exp/ebird_spatial_gnn_baseline.py`.
- The initial support signal is the spatial cell's standardized log count of
  training checklists. This is deliberately simple and non-leaky: it tells the
  residual head how well the local training geography is supported without using
  held-out detections.
- The gate is initialized permissively with `--support-gate-init-bias 2.0`
  (roughly 0.88 after the sigmoid), so it should not erase the residual at the
  start of training. If the model learns that weak-support cells need less
  residual correction, it can shrink them.
- This is a framework-level test. If it helps, later versions can expand the
  gate to include nearest supported training-cell distance, ecology/access
  regime similarity, and species-level support summaries. Those additions should
  still be generic and support-based, not tuned to particular species.

First support-aware coastal-stress run:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10_coastalstress --run-name spatial_gcn_support_gate_cell_frozen_access_h64_l2_z64 --gnn-mode residual --component-mode joint --spatial-channel-mode separated --support-aware-residual cell --support-gate-init-bias 2.0 --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-access-bias-l2 0.001 --frozen-access-embeddings data/ebird/graph_top100_spatial_10x10_coastalstress/access_encoder/access_gcn_h64_l2_z64_cell_embeddings.npy
```

Follow-up diagnostics after the run:

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10_coastalstress --spatial-run-name spatial_gcn_support_gate_cell_frozen_access_h64_l2_z64
python exp/diagnose_ebird_species_block_residuals.py --graph-dir data/ebird/graph_top100_spatial_10x10_coastalstress --run-name spatial_gcn_support_gate_cell_frozen_access_h64_l2_z64 --species "Red-headed Woodpecker" "Northern Rough-winged Swallow" "Green Heron" "Dark-eyed Junco" "Scarlet Tanager" "Swamp Sparrow" "Hairy Woodpecker" "Common Grackle"
```

Initial support-gate result:

- Run name: `spatial_gcn_support_gate_cell_frozen_access_h64_l2_z64`
- Important caveat: the first submitted command did not include
  `--frozen-access-embeddings`, so despite the run name it is an unfrozen-access
  support-gate run and should not be compared directly to the frozen-access
  coastal-stress baseline.
- Metrics:
  - micro AUROC: 0.8862
  - micro AUPRC: 0.5734
  - macro AUROC: 0.8397
  - macro AUPRC: 0.4071
  - ECE: 0.0008
  - max bin error: 0.0075
  - species calibration MAE: 0.0127
- Interpretation: the support gate improved pooled calibration relative to the
  earlier frozen-access coastal-stress run, but ranking was lower. Because the
  access embeddings were not frozen in this run, the next comparison should
  repeat the same support gate with the frozen access encoder included.

Frozen-access support-gate result:

- Run name: `spatial_gcn_support_gate_cell_frozen_access_h64_l2_z64_v2`
- This is the apples-to-apples comparison against the prior frozen-access
  coastal-stress model.
- Metrics:
  - micro AUROC: 0.8858
  - micro AUPRC: 0.5730
  - macro AUROC: 0.8390
  - macro AUPRC: 0.4074
  - ECE: 0.0010
  - max bin error: 0.0045
  - species calibration MAE: 0.0128
- Comparison to prior frozen-access coastal-stress baseline
  (`spatial_gcn_frozen_access_h64_l2_z64`):
  - prior micro/macro AUPRC: 0.5767 / 0.4092
  - support-gated micro/macro AUPRC: 0.5730 / 0.4074
  - prior ECE/species calibration MAE: 0.0012 / 0.0109
  - support-gated ECE/species calibration MAE: 0.0010 / 0.0128
- Interpretation: a cell-density-only support gate is too blunt. It slightly
  improves pooled calibration, but it gives up ranking and species calibration.
  Do not make this the main architecture as-is. The useful lesson is that
  support-aware damping is plausible, but the gate needs richer, more relevant
  support information than total training-checklist density alone.
- Next support-aware variants should be framework-level rather than
  species-specific:
  - local species support: species-specific positive counts or smoothed
    prevalence in nearby training cells
  - regime support: distance to nearest training cell in ecology/access feature
    space
  - uncertainty support: stronger damping where base/residual disagreement is
    high and local training support is weak
  - monotone prior: preserve the portable base model unless support evidence is
    strong enough to justify a spatial residual correction

Species-cell support gate implementation:

- Added `--support-aware-residual species-cell` to
  `exp/ebird_spatial_gnn_baseline.py`.
- This version uses training-only support features for every spatial
  cell/species pair:
  - standardized `log1p` positive detections for that species in that training
    cell
  - standardized smoothed local prevalence logit for that species in that
    training cell
- The gate parameters are shared across species. This is important for the
  overall goal: the model can learn a general rule for when species-specific
  spatial residuals are trustworthy, while each species/cell still supplies its
  own support evidence.
- This is more general than the density-only gate because total checklist
  density measures observer activity, not whether a species' local residual is
  supported. The species-cell gate can damp a residual for a weakly supported
  species even in high-effort cells.
- Diagnostics that reload spatial GNN runs were updated to rebuild these
  training-only support features.

First species-cell support command:

```
python exp/ebird_spatial_gnn_baseline.py --graph-dir data/ebird/graph_top100_spatial_10x10_coastalstress --run-name spatial_gcn_support_gate_species_cell_frozen_access_h64_l2_z64 --gnn-mode residual --component-mode joint --spatial-channel-mode separated --support-aware-residual species-cell --support-gate-init-bias 2.0 --epochs 10 --batch-size 2048 --hidden-dim 128 --hidden-layers 2 --latent-dim 128 --cell-hidden-dim 64 --cell-layers 1 --dropout 0.10 --weight-decay 0.0001 --spatial-grid-size-m 25000 --species-residual-scale sigmoid --species-residual-scale-init 0.10 --species-residual-scale-l2 0.01 --spatial-access-bias-l2 0.001 --frozen-access-embeddings data/ebird/graph_top100_spatial_10x10_coastalstress/access_encoder/access_gcn_h64_l2_z64_cell_embeddings.npy
```

Follow-up diagnostics:

```
python exp/compare_ebird_effort_strata.py --graph-dir data/ebird/graph_top100_spatial_10x10_coastalstress --spatial-run-name spatial_gcn_support_gate_species_cell_frozen_access_h64_l2_z64
python exp/diagnose_ebird_species_block_residuals.py --graph-dir data/ebird/graph_top100_spatial_10x10_coastalstress --run-name spatial_gcn_support_gate_species_cell_frozen_access_h64_l2_z64 --species "Red-headed Woodpecker" "Northern Rough-winged Swallow" "Green Heron" "Dark-eyed Junco" "Scarlet Tanager" "Swamp Sparrow" "Hairy Woodpecker" "Common Grackle"
```

Species-cell support result:

- Run name: `spatial_gcn_support_gate_species_cell_frozen_access_h64_l2_z64`
- Metrics:
  - micro AUROC: 0.8718
  - micro AUPRC: 0.5499
  - macro AUROC: 0.8177
  - macro AUPRC: 0.3849
  - ECE: 0.0117
  - max bin error: 0.0217
  - species calibration MAE: 0.0192
- Comparison to the frozen-access coastal-stress baseline
  (`spatial_gcn_frozen_access_h64_l2_z64`):
  - baseline micro/macro AUPRC: 0.5767 / 0.4092
  - species-cell support micro/macro AUPRC: 0.5499 / 0.3849
  - baseline ECE/species calibration MAE: 0.0012 / 0.0109
  - species-cell support ECE/species calibration MAE: 0.0117 / 0.0192
- Interpretation: this implementation made the residual gating problem worse.
  The general idea remains aligned with the project goal, but direct gating on
  raw cell/species positive support appears too restrictive or too confounded
  with the held-out spatial regime. It likely dampens residual corrections in
  precisely the transfer cases where the model needs a supported ecological
  adjustment.
- Do not promote this run as the current preferred architecture. Use it as a
  diagnostic negative result. Before adding more architecture complexity, inspect
  where it fails by effort stratum, spatial block, and species. If the damage is
  broad, move back to the simpler frozen-access residual model and focus next on
  better validation splits and transferable covariates rather than more gating.

Species-cell support diagnostics:

- Effort-strata diagnostics show broad degradation rather than a localized
  failure mode. Even the "largest gains" over the tabular model are negative:
  - distance `(2,5]`: micro AUPRC delta -0.0077
  - distance `5+`: micro AUPRC delta -0.0087
  - duration `121+`: micro AUPRC delta -0.0098
  - traveling protocol: micro AUPRC delta -0.0137
  - short stationary/zero-distance strata show larger losses, e.g. stationary
    protocol -0.0268 and distance `0` -0.0268.
- Spatial-block losses are also clear:
  - block 79: micro AUPRC delta -0.0526
  - block 48: micro AUPRC delta -0.0297
  - block 65: micro AUPRC delta -0.0122
  - block 17: micro AUPRC delta -0.0107
- Species/block diagnostics show the mechanism: the full model often suppresses
  probabilities far below the portable base model, including on positives.
  Examples:
  - Green Heron, block 17: AUPRC -0.0853, mean probability 0.1317 -> 0.0315
  - Green Heron, block 65: AUPRC -0.0834, mean probability 0.1327 -> 0.0634
  - Red-headed Woodpecker, block 48: AUPRC -0.0834, mean probability 0.2395 -> 0.0407
  - Swamp Sparrow, block 65: AUPRC -0.0595, mean probability 0.1294 -> 0.0761
  - Common Grackle, block 48: AUPRC -0.0483, mean probability 0.3365 -> 0.1402
- Decision: retire `--support-aware-residual species-cell` as a main modeling
  path for now. It is useful evidence that naive support gating can improve some
  calibration errors by suppressing predictions, but it does not support the
  broader goal because it reduces ranking performance and suppresses held-out
  positives across multiple effort regimes and blocks.
- Current preferred architecture returns to the simpler frozen-access residual
  model:
  `spatial_gcn_frozen_access_h64_l2_z64`.
- Next framework work should emphasize:
  - validation across deliberately different spatial regimes
  - clearer comparison to the portable tabular baseline
  - adding transferable covariates or better regime features before adding more
    residual gates
  - if gates return later, make them regularizers or priors rather than direct
    species/cell support suppressors.

Cross-regime validation reset:

- Added `exp/summarize_ebird_split_regime_validation.py` to summarize the
  preferred spatial-GNN run across existing graph split directories without
  retraining.
- Current preferred architecture for this comparison:
  `spatial_gcn_frozen_access_h64_l2_z64`.
- Current cross-regime summary:
  - `primary10x10`: graph micro/macro AUPRC 0.5800 / 0.4151; tabular
    micro/macro AUPRC 0.5725 / 0.4017; deltas +0.0075 / +0.0134; ECE 0.0056;
    74 species improve in AUPRC and 26 decline.
  - `coastalstress`: graph micro/macro AUPRC 0.5767 / 0.4092; tabular
    micro/macro AUPRC 0.5676 / 0.3966; deltas +0.0092 / +0.0125; ECE 0.0012;
    78 species improve in AUPRC and 22 decline.
  - `regime_features`: graph micro/macro AUPRC 0.5762 / 0.4118; both-regime
    tabular micro/macro AUPRC 0.5739 / 0.4056; deltas +0.0023 / +0.0062; ECE
    0.0114; 65 species improve in AUPRC and 35 decline.
- Interpretation: the preferred frozen-access residual model is modestly but
  consistently above the matching tabular baseline across the available split
  regimes. The gain is small, which is important: the current evidence supports
  the GNN residual as a useful add-on, not as a replacement for the portable
  ecological/temporal/effort detector.
- The regime-feature case has the smallest gain and worse calibration, so adding
  coarse regime covariates alone is not clearly better than the simpler `both`
  feature set.
- The per-species comparisons now show that graph gains are widespread but not
  universal. The graph does especially well for some waterbird, open-country,
  and spatially structured species, but loses for several species where the
  tabular model already ranks well or where spatial residuals suppress positives.

Strategic interpretation after architecture search:

- The architecture search has likely reached the point of diminishing returns
  for checklist-level detection prediction. Linear, MLP, link baselines, spatial
  residuals, species graphs, access encoders, environmental edges, and support
  gates all land in the same broad performance band.
- This does not mean the GNN direction failed. It means the current target
  (`checklist x species` detection prediction on complete checklists) is already
  mostly captured by the portable tabular detector, and the GNN adds a modest
  residual improvement.
- For the overall SDM/bias goal, the next gains probably will not come from
  another residual architecture tweak. They should come from stronger structure:
  temporal replication, explicit effort/checklist process modeling, and
  evaluation targets that reward transfer and ecological plausibility rather
  than only all-pairs detection ranking.

Revised next direction:

1. Treat `spatial_gcn_frozen_access_h64_l2_z64` as the current benchmark GNN
   residual framework, not necessarily the final model.
2. Add a locality-time replication dataset that summarizes repeated complete
   checklists at the same or nearby localities through time. This is the most
   general in-dataset source of information for separating occupancy-like
   persistence from effort/detection variation.
3. Add explicit transfer diagnostics beyond stratified spatial holdout:
   mountain -> piedmont/coastal, coastal -> inland, high-effort -> low-effort,
   and year/season transfer.
4. Add ecological plausibility diagnostics:
   species-response curves by canopy/elevation/water/coast distance, phenology
   curves, and comparison against broad known biology for focus species.
5. Keep external structured datasets such as BBS or Atlas as optional anchoring
   data, not a dependency for the core method. They can validate or calibrate
   the framework later, but the method should first exploit complete-checklist
   replication because that is broadly available within eBird.

Target distinction for the pivot:

- Do not silently replace checklist detection with occupancy. The two linked
  targets are:
  - checklist detection: whether species `j` is reported on checklist `i`, given
    effort, time, protocol, observer, location, and environment
  - locality-season availability/occupancy: whether species `j` is plausibly
    present or available at locality `l` during season/year `s`
- The preferred next model family is therefore:
  `availability/occupancy component + checklist detection component`.
- Conceptually:

```
z[j, locality, season] = ecological availability / occupancy
y[j, checklist] ~ detection(z, effort, time, protocol, observer, checklist context)
```

- The current checklist-level detection models remain useful because they
  estimate the observation process. The locality-season dataset adds repeated
  visit structure so that "present but missed" can start to be separated from
  "not available here/now."

Locality-season replication dataset:

- Added `exp/build_ebird_locality_season_dataset.py`.
- The script builds two tables:
  - `locality_seasons.parquet`: one row per locality/season-year group with
    checklist count, date count, observer count, effort summaries, protocol mix,
    environmental medians, coordinates, and adequate-sampling flags
  - `locality_season_species.parquet`: one row per eligible
    locality-season/species pair with `n_checklists`, `n_detections`,
    `n_non_detections`, naive detection rate, effort summaries, and species
    names
- Default locality filter keeps eBird hotspots (`H`) and personal locations
  (`P`) and drops vague/nonstandard locality types. This can be relaxed later
  with `--include-all-locality-types`.
- Default season scheme is `biological-nc`:
  - winter: Dec-Feb, with December assigned to the next winter year
  - spring migration: Mar-Apr
  - early breeding: May-Jun
  - late breeding: Jul-Aug
  - fall migration: Sep-Nov
- Default triplet eligibility is `--min-checklists 3`; the
  `adequate_sampling` flag is stricter and also requires at least two unique
  dates and at least two occupied duration bins.
- A quick count on the full processed dataset suggested the default
  hotspot/personal locality filter yields about 206k locality-season groups,
  about 38.6k with at least three checklists, or roughly 3.86M triplet rows for
  the top 100 species.
- Smoke test with 5,000 checklists and top 20 species passed:
  - 4,998 retained checklists
  - 3,206 locality-season groups
  - 305 eligible locality-seasons
  - 280 adequately sampled locality-seasons
  - 6,100 locality-season/species rows

Build command:

```
python exp/build_ebird_locality_season_dataset.py --processed-dir data/ebird/processed_nc_2020_2023 --output-dir data/ebird/locality_season_top100 --top-species 100 --season-scheme biological-nc --min-checklists 3 --min-dates 2 --min-effort-bins 2 --overwrite
```

Completed full build:

- Retained 661,106 of 661,979 checklists after locality filters.
- Created 206,364 locality-season groups.
- Created 38,639 eligible locality-season groups with at least three
  checklists.
- Flagged 34,328 locality-season groups as adequately sampled under the
  stricter date/effort-variation criteria.
- Created 3,863,900 locality-season/species rows for the top 100 species.
- 1,208,717 locality-season/species rows have at least one detection.

Immediate uses:

- Quantify how much replication is actually available by species, season, and
  locality type.
- Identify locality-seasons with enough varied effort to inform
  occupancy/detection separation.
- Build focus-species diagnostics for strong seasonal migrants such as Wood
  Thrush, plus habitat specialists/generalists used in the spatial-GNN
  diagnostics.
- Provide the data foundation for an explicit hierarchical model where
  locality-season availability is modeled separately from visit-level detection.

Replication-support diagnostics:

- Added `exp/diagnose_ebird_locality_season_replication.py`.
- Output directory:
  `data/ebird/locality_season_top100/diagnostics/replication_support`.
- The diagnostic writes:
  - `overall_summary.csv`
  - `season_summary.csv`
  - `species_replication_summary.csv`
  - `species_season_summary.csv`
  - `effort_support_summary.csv`
  - `focus_species_season_summary.csv`
- Overall support:
  - 206,364 locality-season groups
  - 38,639 eligible locality-season groups
  - 34,328 adequately sampled locality-season groups
  - 3,863,900 locality-season/species rows
  - 1,208,717 positive locality-season/species rows
  - 1,119,416 positive rows are in adequately sampled locality-seasons
- Seasonal support is broad:
  - winter: 10,014 eligible locality-seasons; 271,511 positive triplets
  - spring migration: 7,672 eligible locality-seasons; 269,999 positive triplets
  - early breeding: 7,997 eligible locality-seasons; 264,722 positive triplets
  - late breeding: 5,432 eligible locality-seasons; 156,624 positive triplets
  - fall migration: 7,524 eligible locality-seasons; 245,861 positive triplets
- Focus-species support looks suitable for the pivot:
  - Wood Thrush has strong early-breeding support (2,710 positive
    locality-seasons; 10,822 detections) and essentially no winter support
    (one positive locality-season; one detection), making it a good phenology
    sanity check.
  - Green Heron shows strong breeding-season support and very low winter
    support, also useful for seasonal availability checks.
  - Northern Cardinal and Eastern Towhee have broad year-round support, useful
    for resident/generalist comparisons.
  - Red-headed Woodpecker has support across seasons but lower detection rates,
    useful for testing whether the model can avoid suppressing a species that
    has been difficult for the spatial residual GNN.

Replication diagnostic command:

```
python exp/diagnose_ebird_locality_season_replication.py --dataset-dir data/ebird/locality_season_top100
```

Locality-season binomial baseline:

- Added `exp/ebird_locality_season_baseline.py`.
- This is the first bridge model after the replication dataset. It is not a
  full latent occupancy model yet. It treats each locality-season/species row
  as an aggregated binomial observation:

```
n_detections ~ Binomial(n_checklists, p_species_locality_season)
```

- The baseline compares four models on a held-out season-year:
  - `train_prevalence`: species-specific training prevalence only
  - `availability`: environmental medians plus biological season/year
  - `effort`: locality-season effort summaries
  - `combined`: availability plus effort summaries
- The default split trains on season-years before 2023 and tests on
  season-year 2023. Season-years after 2023 are held out of this comparison,
  which avoids mixing December 2023 winter records into the test year.
- The default dataset is restricted to adequately sampled locality-seasons.
  This keeps the first comparison focused on replicated locality-seasons with
  at least two dates and at least two duration bins.
- Outputs are written to `data/ebird/locality_season_top100/baselines`:
  - `locality_season_baseline_metrics.csv`
  - `locality_season_baseline_species_metrics.csv`
  - `locality_season_baseline_focus_species_season.csv`
  - `locality_season_baseline_summary.json`
- Smoke validation with 20,000 rows and two epochs passed. The expected pattern
  appeared: the prevalence baseline was sensible after species-intercept
  initialization, and the combined model improved weighted BCE and positive
  triplet AUPRC over availability-only and effort-only models. The full run is
  needed before interpreting biological results.

Locality-season baseline command:

```
python exp/ebird_locality_season_baseline.py --dataset-dir data/ebird/locality_season_top100 --epochs 30
```

Completed full locality-season baseline:

- Rows:
  - train: 2,361,700 locality-season/species rows
  - test: 993,200 rows
  - unused: 77,900 rows
- Checklist trials:
  - train: 31,518,300
  - test: 12,771,900
- Held-out detections: 1,961,663
- Held-out observed detection rate: 0.1536
- Overall held-out model comparison:
  - `train_prevalence`: weighted BCE 0.3587; weighted MAE rate 0.1442;
    positive-triplet AUROC/AUPRC 0.7615 / 0.6394; calibration error 0.0039
  - `availability`: weighted BCE 0.3151; weighted MAE rate 0.1171;
    positive-triplet AUROC/AUPRC 0.8474 / 0.7320; calibration error 0.0031
  - `effort`: weighted BCE 0.3390; weighted MAE rate 0.1290;
    positive-triplet AUROC/AUPRC 0.7969 / 0.6820; calibration error 0.0076
  - `combined`: weighted BCE 0.2990; weighted MAE rate 0.1035;
    positive-triplet AUROC/AUPRC 0.8723 / 0.7737; calibration error 0.0045
- Species-level means show the same pattern:
  - `train_prevalence`: mean species AUROC/AUPRC 0.5000 / 0.3544
  - `availability`: mean species AUROC/AUPRC 0.7589 / 0.5513
  - `effort`: mean species AUROC/AUPRC 0.6687 / 0.4537
  - `combined`: mean species AUROC/AUPRC 0.8104 / 0.6305
- The combined model improved species-level AUPRC for 98 of 100 species versus
  availability-only. The largest gains were for Swamp Sparrow, Mallard, Common
  Yellowthroat, White-eyed Vireo, Belted Kingfisher, Black Vulture, Great Blue
  Heron, Indigo Bunting, Northern Rough-winged Swallow, and Wood Duck.
- The only species with AUPRC declines versus availability-only were House
  Sparrow and Red-breasted Nuthatch. These should be checked before treating
  effort effects as uniformly beneficial.
- Interpretation:
  - Availability features, especially season plus environment, explain much
    more locality-season detection-rate structure than effort summaries alone.
  - Effort summaries still add broad complementary information once the
    availability component is present.
  - This is exactly the desired direction for the pivot: first model
    locality-season availability, then let visit/checklist-level effort explain
    detection conditional on availability.
  - The combined model is best for ranking and BCE, but availability-only has
    the lowest aggregate calibration error. This calibration/ranking distinction
    should remain explicit in the next model.
- Focus-species season checks are biologically sensible:
  - Wood Thrush has high early-breeding detection rate (observed 0.1296;
    combined 0.1106) and essentially zero winter detection (observed 0.0000;
    combined 0.0004). Effort-only incorrectly predicts a flat winter rate near
    0.0339.
  - Green Heron has breeding-season peaks and near-zero winter detection
    (observed 0.0008; combined 0.0014), while effort-only again predicts a
    non-biological winter rate near 0.0421.
  - Black-and-white Warbler shows the same pattern: winter observed 0.0071,
    combined 0.0093, effort-only 0.0439.
  - Resident/generalist species such as Northern Cardinal and Eastern Towhee
    retain broad year-round rates, with seasonal peaks and mild underprediction
    in some high-detection seasons.
- Next modeling implication: move from the aggregate binomial bridge to an
  explicit two-component model. The first version should preserve this
  availability/effort separation rather than collapsing everything back into a
  single checklist-level detector.

Two-component checklist detection bridge:

- Added `exp/ebird_locality_season_detection_model.py`.
- This script implements the first explicit bridge from the aggregate
  locality-season baseline to a checklist-level detection model.
- Stage 1 fits an aggregate availability-style model from biological
  season/year and environmental medians:

```
n_detections ~ Binomial(n_checklists, p_available_detectable_locality_season_species)
```

- Stage 2 broadcasts the aggregate locality-season/species availability score
  back to individual complete checklists and fits checklist-level detection
  models with effort and timing features:
  - `train_prevalence`: species train prevalence only
  - `availability_only`: aggregate locality-season/species score only
  - `effort_only`: checklist effort/timing features plus species intercept
  - `two_component`: aggregate availability score plus checklist effort/timing
    features
- The intended interpretation is not "did AUROC move a tiny amount?" The key
  checks are:
  - Does `availability_only` preserve the biologically plausible
    locality-season/phenology structure from the aggregate baseline?
  - Does `two_component` improve checklist-level detection calibration or
    ranking without destroying seasonal plausibility?
  - Do effort effects help focus species and effort strata in ways that are
    consistent with detection rather than observer geography?
- The script writes:
  - `{run_name}_metrics.csv`
  - `{run_name}_species_metrics.csv`
  - `{run_name}_species_delta_vs_availability.csv`
  - `{run_name}_calibration.csv`
  - `{run_name}_focus_species_season.csv`
  - `{run_name}_strata_metrics.csv`
  - `{run_name}_strata_deltas.csv`
  - `{run_name}_county_season_metrics.csv`
  - `{run_name}_county_season_deltas.csv`
  - `{run_name}_county_season_calibration.csv`
  - `{run_name}_focus_species_season_calibration.csv`
  - `{run_name}_focus_species_month.csv`
  - `{run_name}_focus_species_response.csv`
  - `{run_name}_summary.json`
- A small smoke pass validated the joins, label construction, availability
  broadcast, and output writing. The smoke pass is not biologically
  interpretable because it used one epoch and a small checklist subset.

Two-component checklist detection command:

```
python exp/ebird_locality_season_detection_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --availability-epochs 30 --detection-epochs 30
```

If runtime is too long, the first acceptable full-ish diagnostic run is:

```
python exp/ebird_locality_season_detection_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --availability-epochs 10 --detection-epochs 20 --run-name two_component_checklist_detection_e10_d20
```

Completed first two-component checklist detection run:

- Command:

```
python exp/ebird_locality_season_detection_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --availability-epochs 10 --detection-epochs 20 --run-name two_component_checklist_detection_e10_d20
```

- Retained 450,699 of 661,979 checklists after locality-season filters.
- Checklist rows:
  - train: 315,183
  - test: 127,719
  - unused: 7,797
- Checklist/species pairs:
  - train: 31,518,300
  - test: 12,771,900
- Held-out detections: 1,961,663
- Held-out observed detection rate: 0.1536
- Overall held-out checklist-level metrics:
  - `train_prevalence`: BCE 0.3587; micro AUROC/AUPRC 0.7764 / 0.4209;
    ECE 0.0039; calibration error 0.0039
  - `availability_only`: BCE 0.3154; micro AUROC/AUPRC 0.8480 / 0.5077;
    ECE 0.0052; calibration error 0.0031
  - `effort_only`: BCE 0.3168; micro AUROC/AUPRC 0.8457 / 0.5154;
    ECE 0.0070; calibration error 0.0069
  - `two_component`: BCE 0.2910; micro AUROC/AUPRC 0.8761 / 0.5767;
    ECE 0.0066; calibration error 0.0065
- Mean species-level metrics:
  - `train_prevalence`: mean species AUROC/AUPRC 0.5000 / 0.1535;
    mean species calibration error 0.0085
  - `availability_only`: mean species AUROC/AUPRC 0.7537 / 0.2996;
    mean species calibration error 0.0090
  - `effort_only`: mean species AUROC/AUPRC 0.7525 / 0.3017;
    mean species calibration error 0.0099
  - `two_component`: mean species AUROC/AUPRC 0.8154 / 0.3881;
    mean species calibration error 0.0108
- Species-level AUPRC:
  - `two_component` improved over `availability_only` for 99 of 100 species.
  - Largest AUPRC gains were for White-eyed Vireo, Eastern Wood-Pewee, Common
    Yellowthroat, Ruby-crowned Kinglet, Turkey Vulture, American Redstart,
    Eastern Phoebe, Indigo Bunting, Blue-gray Gnatcatcher, and Swamp Sparrow.
  - The only AUPRC decline was Red-breasted Nuthatch (-0.0094), so this species
    remains the main case to inspect for overcorrection or weak support.
- Focus-species seasonal plausibility remains mostly sound:
  - Wood Thrush early breeding is captured almost exactly by `two_component`
    (observed 0.1296; predicted 0.1295) and winter remains near zero (observed
    0.0000; predicted 0.0008).
  - Green Heron winter remains near zero (observed 0.0008; predicted 0.0016),
    while breeding-season rates remain high.
  - Black-and-white Warbler winter is corrected downward relative to
    `availability_only` and `effort_only` (observed 0.0071; `two_component`
    0.0064).
  - Northern Cardinal and Eastern Towhee retain broad year-round detection
    patterns, though `two_component` mildly underpredicts some resident-season
    rates.
- Interpretation:
  - This is the strongest evidence so far that the pivot is productive. The
    aggregate availability score carries biological/seasonal structure, and
    checklist-level effort/timing adds substantial detection-ranking signal.
  - The result is not just another spatial-GNN architecture tweak; it explicitly
    separates locality-season availability from checklist-level detection while
    still using complete-checklist non-detections.
  - Calibration is acceptable but not uniformly best. `availability_only` has
    the lowest pooled calibration error, while `two_component` has much better
    BCE/ranking. Future work should inspect calibration by probability bin,
    species, effort strata, and season rather than relying on pooled ECE alone.
- Script update after this run: the script now also writes calibration-bin
  outputs and species AUPRC deltas versus availability-only, and prints mean
  species-level metrics in the console for future runs.

Two-component diagnostic script:

- Added `exp/diagnose_ebird_locality_season_detection.py`.
- This reads saved outputs from the two-component checklist detection bridge and
  summarizes:
  - overall metric deltas for `two_component` versus `availability_only`,
    `effort_only`, and `train_prevalence`
  - mean species-level AUROC/AUPRC/calibration
  - species AUPRC gains/losses versus availability-only
  - worst predicted-probability calibration bins
  - focus-species/season calibration changes versus availability-only
  - effort/locality stratum deltas and worst stratum calibration when
    `_strata_metrics.csv` and `_strata_deltas.csv` are available
  - county-season AUPRC/ECE deltas and worst county-season probability bins when
    the county-season calibration CSVs are available
  - focus-species season probability-bin calibration when
    `_focus_species_season_calibration.csv` is available
- This script does not retrain the model and does not reconstruct
  checklist-level predictions. It is a fast diagnostic layer from saved summary
  outputs. County-season and focus-season calibration bins are available after
  rerunning the detection model script with the current version.

Diagnostic command:

```
python exp/diagnose_ebird_locality_season_detection.py --model-dir data/ebird/locality_season_top100/detection_models --run-name two_component_checklist_detection_e10_d20 --top 10
```

Direct effort/locality stratum diagnostics were added to the two-component
training script after this saved-output diagnostic pass. These are computed
inside the training run while checklist-level predictions are already in memory,
rather than saving large prediction matrices for a separate post-processing
step.

Rerun command to add the stratum outputs:

```
python exp/ebird_locality_season_detection_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --availability-epochs 10 --detection-epochs 20 --run-name two_component_checklist_detection_e10_d20
```

New output files from that rerun:

- `two_component_checklist_detection_e10_d20_strata_metrics.csv`
- `two_component_checklist_detection_e10_d20_strata_deltas.csv`
- updated `two_component_checklist_detection_e10_d20_summary.json` with stratum
  row counts and the `stratum_min_checklists` setting

The strata currently include protocol, duration, effort distance, observer
count, start-hour bin, biological season, locality type, county,
locality-season checklist-count bin, locality-season date-count bin, and
locality-season effort-bin count. This is intended to answer where the
two-component availability/detection bridge helps because effort explains
missed detections, and where it overcorrects relative to availability-only or
effort-only baselines.

Completed first effort/locality stratum diagnostics:

- The rerun reproduced the previous overall and species-level two-component
  results exactly, then added stratum metrics and deltas.
- Across the reported large strata, `two_component` improved micro-AUPRC over
  `availability_only` in every stratum:
  - minimum stratum micro-AUPRC delta: +0.0177
  - maximum stratum micro-AUPRC delta: +0.0943
  - negative stratum deltas: 0 of 66
- The largest AUPRC gains over availability-only occurred in several county
  strata and in broad effort/season strata:
  - Person, Craven, Forsyth, Gaston, Henderson, Cumberland, Wake, and Orange
    counties
  - checklists with 3+ observers
  - fall migration and winter
  - locality-seasons with 26-50 checklists
- The smallest gains were still positive, not true losses. They included
  Jackson County, short-duration checklists, several duration bins, Carteret,
  Yancey, Transylvania, the `(2,5]` and `(0.5,2]` distance bins, and Beaufort.
  The console label was updated after this run so future output says
  "smallest gains" unless an actual negative stratum delta exists.
- Calibration changed differently by stratum:
  - Strong calibration improvements versus availability-only occurred for long
    and very short duration strata, long-distance traveling strata, evening and
    early-morning strata, 3+ observers, Stationary protocol, and zero-distance
    checklists.
  - The largest calibration degradations were concentrated in specific counties
    and spring migration. Beaufort, Guilford, Yancey, Wake, Forsyth, Craven,
    Dare, and Haywood should be treated as county-level calibration inspection
    targets, not as evidence that the two-component bridge failed overall.
- Worst `two_component` stratum calibration was mostly county-specific:
  Craven, Transylvania, Yancey, Beaufort, Person, Cumberland, Guilford, Wake,
  Henderson, Watauga, and Gaston. A low-replication locality-season date bin
  (`1-2` dates) also appeared, which reinforces that replication depth matters.
- Interpretation:
  - This is a strong diagnostic result for the pivot. The two-component model is
    not only improving pooled metrics; it improves ranking across all tested
    effort/locality strata.
  - The remaining issue is calibration locality/regime heterogeneity, not a lack
    of effort/detection signal.
  - The next step should be calibration and component diagnostics for the
    two-component bridge, not another broad spatial-GNN architecture variant.

Completed first two-component diagnostics:

- Overall deltas for `two_component` versus `availability_only`:
  - BCE improved by -0.0243.
  - Micro AUROC improved by +0.0281.
  - Micro AUPRC improved by +0.0690.
  - ECE worsened slightly by +0.0015.
  - Pooled calibration error worsened slightly by +0.0034.
  - Max bin error improved substantially because `availability_only` had a very
    small high-probability bin with severe overprediction.
- Overall deltas for `two_component` versus `effort_only`:
  - BCE improved by -0.0257.
  - Micro AUROC improved by +0.0305.
  - Micro AUPRC improved by +0.0613.
  - ECE and pooled calibration error were slightly better.
- Species-level AUPRC versus availability-only:
  - improved for 99 species
  - declined for one species
  - mean species AUPRC delta was +0.0885
  - Red-breasted Nuthatch remains the only negative species-level AUPRC case
    (-0.0094)
- Worst calibration bins:
  - `availability_only` has a severe high-probability calibration issue in the
    0.9-1.0 bin: mean predicted 0.9328 versus observed 0.6651, but only 1,072
    pairs are in that bin.
  - `two_component` has its largest bin errors in much larger mid-probability
    bins: 0.3-0.4, 0.4-0.5, and 0.5-0.6. It tends to underpredict observed rates
    in these bins by about 0.02.
  - This suggests that `two_component` improves ranking and removes the most
    extreme high-bin issue, but still needs calibration diagnostics beyond a
    single pooled ECE value.
- Focus species/season:
  - `two_component` most improved absolute error for Double-crested Cormorant
    early breeding, Black-and-white Warbler fall migration/winter, and Wood
    Thrush winter.
  - `two_component` most worsened absolute error for Northern Cardinal winter,
    Eastern Towhee late breeding/winter, Double-crested Cormorant winter/late
    breeding, and Wood Thrush spring migration.
  - Several worsened focus rows are resident/generalist or common-waterbird
    cases where availability-only was already close, so the effort correction may
    be over-adjusting some high-prevalence seasonal rates.
- Interpretation:
  - The bridge is on the right track. The main value is not only a metric bump;
    it is that availability and detection are now separable enough to diagnose
    where effort/timing helps versus where it overcorrects.
  - The next modeling step should be diagnostic-driven calibration and
    focus-species inspection, not another broad architecture sweep.

Completed county-season and focus-season calibration diagnostics:

- The repeated 10/20 two-component run reproduced the earlier headline metrics:
  `two_component` held-out BCE 0.2910, micro AUROC 0.8762, micro AUPRC 0.5767,
  and ECE 0.0066. This suggests the current run is stable enough for
  diagnostics without immediately moving to a longer 30/30 epoch run.
- The diagnostic script now filters ranked probability-bin calibration rows
  with `--min-bin-pairs` (default `100`). This matters because the unfiltered
  focus-species and county-season bin tables were dominated by tiny bins with
  only a handful of checklist/species pairs.
- The filtered rerun successfully excluded tiny probability bins from the
  ranked tables. Stable county-season issues remain concentrated in Dare
  high-probability bins, Carteret spring migration, and Craven winter, while
  stable focus-species issues include Eastern Meadowlark overprediction, Great
  Egret high-probability overprediction, and selected underprediction for
  Double-crested Cormorant, Green Heron, and Black-and-white Warbler.
- County-season ranking remained broadly positive:
  - county-season AUPRC improved in all reported county-season groups
  - minimum county-season AUPRC gain was about +0.021
  - maximum county-season AUPRC gain was about +0.112
  - largest gains included Craven winter, Henderson winter, Forsyth late
    breeding/fall/winter, Cumberland winter, Wake winter, Guilford winter, and
    Watauga spring migration
- Remaining county-season concerns are calibration pockets, not ranking:
  - worst overall county-season ECE included Chatham early breeding, Watauga
    early breeding, Henderson fall migration, Craven winter, Guilford spring
    migration/early breeding, Wake spring/winter, Brunswick early breeding, and
    Dare early breeding
  - after filtering to bins with at least 100 pairs, stable high-probability
    overprediction is concentrated in Dare, especially winter/spring/fall
    high-probability bins; Carteret spring also appears
  - Craven winter shows an opposite issue in one stable bin: underprediction
    around the 0.6-0.7 probability range
- Focus-species stable probability-bin issues:
  - Eastern Meadowlark is overpredicted in several stable early/spring/fall
    seasonal bins
  - Great Egret has high-probability seasonal overprediction in fall/late
    breeding
  - Double-crested Cormorant late breeding and Green Heron fall show
    underprediction in stable bins
  - Wood Thrush still passes the important sanity check of near-zero winter
    detection, but spring migration remains underpredicted
- Interpretation:
  - The two-component bridge is now clearly more useful than the one-component
    baselines for ranking and stratified detection prediction.
  - The next work should not chase small pooled metric gains. It should test
    whether the availability component and detection component produce
    biologically plausible seasonal and environmental responses for focus
    species.
  - Post-hoc calibration or calibration-aware loss may be useful later, but it
    should follow response/phenology diagnostics so calibration fixes do not
    hide ecological misspecification.

Completed first monthly phenology and environmental-response diagnostic review:

- The updated model run wrote:
  - 120 monthly focus-species rows: 10 species x 12 months
  - 320 environmental-response rows: 10 species x 4 covariates x 8 bins
- Monthly phenology:
  - the two-component model had lower checklist-weighted monthly error than
    availability-only for Eastern Meadowlark, Green Heron, Great Egret, Wood
    Thrush, and Black-and-white Warbler
  - availability-only remained better for Red-headed Woodpecker, House Sparrow,
    Double-crested Cormorant, Eastern Towhee, and Northern Cardinal, although
    the Northern Cardinal difference was small in aggregate
  - the two-component model corrected useful seasonal errors such as Green Heron
    and Wood Thrush near-zero months and Black-and-white Warbler late fall
  - its largest monthly degradations were mostly level shifts for common
    residents or waterbirds, including Eastern Towhee summer/winter, Northern
    Cardinal winter, and Double-crested Cormorant winter
- Environmental response:
  - across focus species, response-bin probability error was slightly worse for
    the two-component model than availability-only for canopy, water distance,
    and coast distance, and approximately tied for elevation
  - response shape ordering was nevertheless usually retained; mean Spearman
    agreement improved for canopy and water distance but weakened somewhat for
    coastline distance and elevation
  - strong two-component improvements included Double-crested Cormorant canopy
    and elevation, Great Egret elevation, and Red-headed Woodpecker elevation
    and coastline distance
  - the largest degradations were concentrated in Northern Cardinal across
    multiple covariates, Eastern Towhee canopy/coast distance, and smaller
    probability-level shifts for Wood Thrush despite strong shape agreement
- Interpretation:
  - the detection component is adding real ranking signal without generally
    destroying ecological ordering
  - the main failure is probability-level overcorrection or undercorrection for
    some species/regimes, not wholesale ecological response reversal
  - this supports testing shrinkage or partial pooling of species-specific
    detection effects rather than replacing the framework or returning to a
    broad GNN architecture search
  - these are marginal binned response diagnostics, not causal partial
    dependence curves; season, geography, and correlated habitat covariates can
    still confound their shapes
- Visual review of the generated plots confirmed this interpretation:
  - the two-component curve generally follows the availability-only curve rather
    than reversing its shape
  - Northern Cardinal shows a systematic downward shift from availability-only
    across elevation, water-distance, and coastline-distance bins
  - Eastern Towhee shows its largest degradation for canopy and coastline
    response levels
  - Wood Thrush retains the expected increasing canopy/elevation/distance
    response shapes, but the two-component correction suppresses the highest
    predicted habitat bins too strongly
  - Double-crested Cormorant is the clearest focus-species case where the
    two-component environmental-response levels improve over availability-only
- This is sufficient evidence to test neutral-point shrinkage of the
  checklist-level correction before changing model families.

Plausibility plotting command:

```
python exp/plot_ebird_locality_season_plausibility.py --model-dir data/ebird/locality_season_top100/detection_models --run-name two_component_checklist_detection_e10_d20
```

The script writes:

- `phenology_overview.png`
- one four-panel environmental-response plot per focus species
- `environmental_response_mae_delta.png`
- `phenology_summary.csv`
- `environmental_response_summary.csv`
- `metadata.json`

under:

`data/ebird/locality_season_top100/detection_models/diagnostics/plausibility/two_component_checklist_detection_e10_d20`

Completed first combined detection-shrinkage experiment:

- Run:
  `two_component_checklist_detection_shrink_r0p01_a0p01`
- Penalties:
  - residual/intercept/effort L2: `0.01`
  - availability-weight-to-one L2: `0.01`
- Pooled changes versus the unregularized two-component run:
  - BCE: +0.0008, slightly worse
  - micro AUROC: -0.0010
  - micro AUPRC: -0.0012
  - ECE: -0.0022, improving from 0.0066 to 0.0044
  - max bin error: -0.0047, improving from 0.0230 to 0.0182
  - pooled mean-rate calibration error: -0.0025
- Mean species changes:
  - AUROC: -0.0019
  - AUPRC: -0.0023
  - calibration error: -0.0008
- Species effects were heterogeneous:
  - Red-breasted Nuthatch AUPRC improved by +0.0069, reducing its deficit versus
    availability-only from -0.0094 to -0.0024, although its species calibration
    error worsened slightly
  - Gray Catbird and Red-eyed Vireo also gained AUPRC
  - larger AUPRC losses included Yellow-billed Cuckoo (-0.0202),
    Golden-crowned Kinglet (-0.0131), Acadian Flycatcher (-0.0125), Hermit
    Thrush (-0.0108), Bufflehead (-0.0107), and Hooded Merganser (-0.0100)
- Monthly phenology:
  - overall focus-species weighted monthly MAE worsened from 0.01759 to 0.01799
  - Black-and-white Warbler, Red-headed Woodpecker, Eastern Towhee, House
    Sparrow, and Wood Thrush improved slightly
  - Green Heron and Double-crested Cormorant worsened the most
- Environmental responses:
  - mean weighted response error improved slightly for all four covariates
  - the strongest improvements were Black-and-white Warbler canopy and Wood
    Thrush water/elevation/coast responses
  - shape agreement was mostly stable, but some Green Heron response shapes
    weakened
- Learned regularized parameter summary:
  - species intercept RMS: 0.1456
  - effort-weight RMS: 0.1923
  - mean availability multiplier: 1.0675
  - availability-multiplier deviation RMS from one: 0.0913
- Interpretation:
  - the combined penalty demonstrates that calibration can improve without a
    large pooled ranking collapse
  - it is not yet preferable overall because monthly phenology and individual
    species effects are mixed
  - because two penalties changed simultaneously, the next experiment should
    isolate residual/effort shrinkage from availability-weight shrinkage

Saved-run comparison utility:

```
python exp/compare_ebird_locality_season_runs.py --run two_component_checklist_detection_e10_d20 --run two_component_checklist_detection_shrink_r0p01_a0p01
```

This writes pooled, species, phenology, and environmental-response comparison
CSVs under:

`data/ebird/locality_season_top100/detection_models/diagnostics/run_comparisons`

Completed residual-only shrinkage ablation:

- Run:
  `two_component_checklist_detection_shrink_r0p01_a0`
- Penalties:
  - residual/intercept/effort L2: `0.01`
  - availability-weight-to-one L2: `0`
- Changes versus the unregularized two-component run:
  - BCE: +0.0008
  - micro AUROC: -0.0011
  - micro AUPRC: -0.0014
  - ECE: -0.0022, improving from 0.0066 to 0.0044
  - max bin error: -0.0016, improving from 0.0230 to 0.0213
  - mean species AUPRC: -0.0023
  - mean species calibration error: -0.0008
- Changes versus the combined `0.01 / 0.01` run were very small:
  - micro AUPRC was lower by about 0.0002
  - ECE was effectively unchanged
  - max bin error was worse by about 0.0031
  - mean focus-species monthly MAE was worse by about 0.00005
  - mean environmental-response MAE was better by only about 0.00001
- The species gains/losses were also nearly the same as the combined run.
  Red-breasted Nuthatch again gained about +0.0069 AUPRC, while
  Yellow-billed Cuckoo, Golden-crowned Kinglet, Acadian Flycatcher, Hermit
  Thrush, Bufflehead, and Hooded Merganser remained among the largest losses.
- Parameter comparison:
  - residual-only effort-weight RMS: 0.1951
  - combined effort-weight RMS: 0.1923
  - residual-only availability multiplier mean/deviation: 1.0853 / 0.1178
  - combined availability multiplier mean/deviation: 1.0675 / 0.0913
- Interpretation:
  - residual shrinkage is responsible for nearly all of the broad ECE
    improvement and most of the species-ranking cost
  - availability-weight shrinkage contributes a small benefit to max-bin
    calibration and pooled ranking when combined with residual shrinkage
  - neither run is a clear new default because both slightly worsen aggregate
    focus-species monthly phenology
  - the next clean experiment is availability-weight-only shrinkage

The initial three-run comparison failed only while writing output because the
automatic filename exceeded a Windows path-component limit. The comparison
script now uses a compact hashed name by default and also accepts
`--comparison-name`. The repaired comparison completed successfully.

Completed availability-weight-only shrinkage ablation:

- Run:
  `two_component_checklist_detection_shrink_r0_a0p01`
- Penalties:
  - residual/intercept/effort L2: `0`
  - availability-weight-to-one L2: `0.01`
- Changes versus the unregularized two-component run:
  - BCE: +0.00014
  - micro AUROC: -0.00019
  - micro AUPRC: +0.00007
  - ECE: -0.00006
  - max bin error: -0.00132, from 0.02295 to 0.02163
  - mean species AUPRC: +0.00007
  - mean species calibration error: -0.00004
- Plausibility changes were also very small:
  - mean focus-species monthly MAE increased by 0.00003
  - mean environmental-response MAE improved by 0.00011
  - mean response-shape Spearman declined by 0.00179
- Species effects were much smaller than under residual shrinkage:
  - the largest AUPRC gain was Gray Catbird at +0.00555
  - the largest AUPRC loss was Hooded Merganser at -0.00282
  - Red-breasted Nuthatch gained +0.00131 but still remained below its
    availability-only comparator
- Learned parameter summary:
  - species intercept RMS: 0.7088
  - effort-weight RMS: 0.2354
  - mean availability multiplier: 0.9578
  - availability-multiplier deviation RMS from one: 0.0976
- Four-run comparison:
  - unregularized micro AUPRC / ECE: 0.57670 / 0.00664
  - combined `0.01 / 0.01`: 0.57549 / 0.00442
  - residual-only `0.01`: 0.57528 / 0.00443
  - availability-only `0.01`: 0.57677 / 0.00658
- Interpretation:
  - availability-weight shrinkage alone is effectively a no-op at this strength
  - residual shrinkage produces nearly all of the pooled calibration gain and
    nearly all of the associated ranking and phenology cost
  - the combined penalty's slightly better max-bin behavior does not justify
    treating availability-weight shrinkage as the primary control
  - the next clean experiment is weaker residual-only shrinkage, not a stronger
    availability penalty

Four-run comparison command:

```
python exp/compare_ebird_locality_season_runs.py --run two_component_checklist_detection_e10_d20 --run two_component_checklist_detection_shrink_r0p01_a0p01 --run two_component_checklist_detection_shrink_r0p01_a0 --run two_component_checklist_detection_shrink_r0_a0p01 --comparison-name shrinkage_ablation_001
```

The comparison outputs are under:

`data/ebird/locality_season_top100/detection_models/diagnostics/run_comparisons`

Completed weaker residual-only shrinkage experiment:

- Run:
  `two_component_checklist_detection_shrink_r0p0025_a0`
- Penalties:
  - residual/intercept/effort L2: `0.0025`
  - availability-weight-to-one L2: `0`
- Changes versus the unregularized two-component run:
  - BCE: approximately +0.00041
  - micro AUROC: approximately -0.00056
  - micro AUPRC: -0.00067, from 0.57670 to 0.57603
  - ECE: -0.00137, from 0.00664 to 0.00527
  - max bin error: approximately -0.00050, from 0.02295 to 0.02245
  - mean species AUPRC: -0.00106, from 0.38812 to 0.38706
  - mean species calibration error: -0.00047, from 0.01083 to 0.01035
- Relative to residual L2 `0.01`, the `0.0025` run retained:
  - about 62% of the pooled ECE improvement
  - about 47% of the pooled AUPRC loss
  - about 45% of the mean-species AUPRC loss
  - about 57% of the mean-species calibration improvement
- Plausibility comparison:
  - mean focus-species monthly MAE increased by 0.00025 versus 0.00046 at
    residual L2 `0.01`
  - mean environmental-response MAE improved by 0.00038 versus 0.00041 at
    residual L2 `0.01`
  - mean response-shape Spearman declined by 0.00476 versus 0.00595 at
    residual L2 `0.01`
- Species-level changes:
  - 26 species gained AUPRC and 74 lost AUPRC relative to the unregularized run
  - the median species change was only -0.00029
  - Red-breasted Nuthatch improved by +0.00488
  - the largest losses were smaller than at `0.01`, including
    Yellow-billed Cuckoo (-0.01291), Acadian Flycatcher (-0.00879), Hooded
    Merganser (-0.00801), Bufflehead (-0.00742), and Golden-crowned Kinglet
    (-0.00724)
- Learned parameter summary:
  - species intercept RMS: 0.3175
  - effort-weight RMS: 0.2088
  - mean availability multiplier: 1.0379
  - availability-multiplier deviation RMS from one: 0.0968
- Interpretation:
  - `0.0025` is meaningfully different from the unregularized model and
    substantially less destructive than `0.01`
  - calibration improvement scales nonlinearly with the penalty; a relatively
    weak penalty recovers more than half of the ECE benefit
  - environmental-response level error improves almost as much as at `0.01`,
    while ranking and phenology losses are materially smaller
  - max-bin calibration changes little, so pooled ECE should not be the only
    criterion used to select the final operating point
  - one midpoint run at `0.005` is justified; further scalar L2 search after
    that would have diminishing scientific value

Residual-strength comparison command:

```
python exp/compare_ebird_locality_season_runs.py --run two_component_checklist_detection_e10_d20 --run two_component_checklist_detection_shrink_r0p01_a0 --run two_component_checklist_detection_shrink_r0p0025_a0 --comparison-name residual_shrinkage_strength
```

Completed midpoint residual-only shrinkage experiment:

- Run:
  `two_component_checklist_detection_shrink_r0p005_a0`
- Penalties:
  - residual/intercept/effort L2: `0.005`
  - availability-weight-to-one L2: `0`
- Changes versus the unregularized two-component run:
  - micro AUPRC: -0.00104, from 0.57670 to 0.57566
  - ECE: -0.00180, from 0.00664 to 0.00484
  - mean species AUPRC: -0.00168, from 0.38812 to 0.38644
  - mean species calibration error: -0.00064, from 0.01083 to 0.01019
  - focus-species monthly MAE: +0.00035
  - environmental-response MAE: -0.00041
  - response-shape Spearman: -0.00476
- Relative to residual L2 `0.01`, the `0.005` run retained:
  - about 81% of the ECE improvement
  - about 73% of the pooled AUPRC loss
  - about 71% of the mean-species AUPRC loss
  - about 77% of the mean-species calibration improvement
  - about 75% of the focus-species phenology degradation
- Species effects:
  - 25 species gained AUPRC and 75 lost AUPRC relative to the unregularized run
  - median species AUPRC change was -0.00058
  - Red-breasted Nuthatch improved by +0.00591
  - larger losses included Yellow-billed Cuckoo (-0.01586), Acadian
    Flycatcher (-0.01078), Golden-crowned Kinglet (-0.01060), Bufflehead
    (-0.00917), and Hooded Merganser (-0.00911)
- Learned parameter summary:
  - species intercept RMS: 0.2098
  - effort-weight RMS: 0.2020
  - mean availability multiplier: 1.0652
  - availability-multiplier deviation RMS from one: 0.1062
- Interpretation:
  - the four residual strengths form a smooth Pareto sequence rather than
    identifying a uniquely best scalar penalty
  - `0.005` gains only 0.00043 additional ECE improvement over `0.0025`, while
    losing another 0.00037 pooled AUPRC, 0.00062 mean species AUPRC, and
    increasing monthly phenology MAE by another 0.00010
  - `0.0025` is therefore the preferred conservative regularized sensitivity
    run, not a replacement for the unregularized ranking benchmark
  - further scalar L2 tuning would optimize this NC/top-100 fit without solving
    the structural separation problem

Complete residual-strength comparison command:

```
python exp/compare_ebird_locality_season_runs.py --run two_component_checklist_detection_e10_d20 --run two_component_checklist_detection_shrink_r0p0025_a0 --run two_component_checklist_detection_shrink_r0p005_a0 --run two_component_checklist_detection_shrink_r0p01_a0 --comparison-name residual_shrinkage_strength_complete
```

Partial-pooling implementation:

- `exp/ebird_locality_season_detection_model.py` now accepts:
  `--two-component-effort-mode species|shared|partial`.
- `species` preserves the existing independent species-specific effort
  coefficients.
- `shared` estimates one checklist-effort response applied to every species.
- `partial` estimates:
  - one shared effort response
  - zero-mean species-specific deviations around that response
- Under `partial`, `--two-component-residual-l2` shrinks the species intercept
  corrections and species effort deviations but does not shrink the shared
  effort response. This is the intended partial-pooling behavior: common
  detection effects remain learnable while unsupported species differences are
  pulled toward the common response.
- Existing commands remain unchanged because the default mode is `species`.

Completed partial-pooling result:

- Run: `two_component_checklist_detection_partial_r0p0025`
- Command:

```
python exp/ebird_locality_season_detection_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --availability-epochs 10 --detection-epochs 20 --run-name two_component_checklist_detection_partial_r0p0025 --two-component-effort-mode partial --two-component-residual-l2 0.0025
```

- Held-out checklist-level metrics:
  - BCE: 0.29146
  - micro AUROC: 0.87561
  - micro AUPRC: 0.57601
  - ECE: 0.00575
  - max bin error: 0.02286
- Mean species metrics:
  - AUROC: 0.81437
  - AUPRC: 0.38711
  - calibration error: 0.01050
- Relative to the unregularized two-component run:
  - micro AUPRC: -0.00068
  - ECE: -0.00089
  - mean species AUPRC: -0.00101
  - mean species calibration error: -0.00033
- Relative to species-specific residual-only shrinkage at `0.0025`, partial
  pooling is effectively tied on ranking but has weaker pooled and species
  calibration:
  - partial ECE 0.00575 vs 0.00527 for species-specific `0.0025`
  - partial mean species AUPRC 0.38711 vs 0.38706
  - partial mean species calibration error 0.01050 vs 0.01035
- Interpretation: partial pooling is not a new preferred model yet. It confirms
  that weak regularization can be applied without collapsing the detector, but
  it does not materially improve on simple species-specific weak shrinkage.
  The next useful test is a fully shared effort response to determine whether
  species-specific effort deviations are necessary at all.

Partial-pooling comparison command:

```
python exp/compare_ebird_locality_season_runs.py --run two_component_checklist_detection_e10_d20 --run two_component_checklist_detection_shrink_r0p0025_a0 --run two_component_checklist_detection_partial_r0p0025 --comparison-name partial_pooling_first
```

Completed shared-effort ablation:

- Run: `two_component_checklist_detection_shared_r0p0025`
- Command:

```
python exp/ebird_locality_season_detection_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --availability-epochs 10 --detection-epochs 20 --run-name two_component_checklist_detection_shared_r0p0025 --two-component-effort-mode shared --two-component-residual-l2 0.0025
```

- Held-out checklist-level metrics:
  - BCE: 0.29831
  - micro AUROC: 0.86805
  - micro AUPRC: 0.56023
  - ECE: 0.00714
  - max bin error: 0.02699
- Mean species metrics:
  - AUROC: 0.80053
  - AUPRC: 0.36940
  - calibration error: 0.01071
- Relative to the unregularized two-component run:
  - micro AUPRC: -0.01647
  - ECE: +0.00050
  - mean species AUPRC: -0.01872
  - mean species calibration error: -0.00012
- Relative to `availability_only`, shared effort still adds detection signal
  (micro AUPRC 0.56023 vs 0.50772; mean species AUPRC 0.36940 vs 0.29961),
  but it gives up too much of the two-component gain.
- Largest species AUPRC losses versus the unregularized two-component run were
  large and biologically diverse:
  - House Finch: -0.12967
  - Ruby-throated Hummingbird: -0.12340
  - Golden-crowned Kinglet: -0.06911
  - Turkey Vulture: -0.05769
  - American Redstart: -0.04953
  - Acadian Flycatcher: -0.04846
  - Eastern Wood-Pewee: -0.04808
  - Swamp Sparrow: -0.04578
  - White-eyed Vireo: -0.04570
  - Yellow-billed Cuckoo: -0.04338
- A few species gained under shared effort, including Red-breasted Nuthatch
  (+0.02498), Brown-headed Nuthatch (+0.01147), and Gray Catbird (+0.00969),
  but these gains are not enough to offset broad losses.
- Interpretation: fully shared effort is too restrictive. It confirms that
  checklist effort effects are not merely a single global observer-process
  correction; species differ in how duration, observers, distance, time, and
  related checklist features affect report probability. The general framework
  should retain species-specific detection responses, with optional weak
  shrinkage or partial pooling as sensitivity checks rather than replacements.

Effort-pooling comparison command:

```
python exp/compare_ebird_locality_season_runs.py --run two_component_checklist_detection_e10_d20 --run two_component_checklist_detection_shrink_r0p0025_a0 --run two_component_checklist_detection_partial_r0p0025 --run two_component_checklist_detection_shared_r0p0025 --comparison-name effort_pooling_ablation
```

Latent repeated-visit model implementation:

- Added `exp/ebird_locality_season_latent_model.py`.
- This script is the first direct repeated-visit availability/detection model.
  It does not broadcast a fixed aggregate availability score into a second-stage
  detector. Instead, it jointly optimizes:
  - locality-season/species availability probability \(\psi_{j,l,s}\)
  - checklist/species detection probability \(p_{j,i}\), conditional on
    availability
- For a locality-season/species group with at least one detection, the
  likelihood is:

```
log psi + sum_i log Bernoulli(y_i | p_i)
```

- For a group with no detections, the likelihood is:

```
log((1 - psi) + psi * product_i(1 - p_i))
```

- This directly addresses the bridge model's main limitation: a zero-detection
  locality-season is no longer forced to be low availability before the model
  has considered repeated missed detections under the visit effort distribution.
- The first implementation is intentionally plain:
  - availability features: biological season/year plus environmental covariates
  - detection features: checklist effort, timing, protocol, and species-specific
    detection responses
  - no GNN, no spatial residual, no post-hoc calibration
- Outputs:
  - `{run_name}_metrics.csv`
  - `{run_name}_species_metrics.csv`
  - `{run_name}_availability_metrics.csv`
  - `{run_name}_availability_species_metrics.csv`
  - `{run_name}_focus_species_season.csv`
  - `{run_name}_focus_species_availability_season.csv`
  - `{run_name}_latent_detection_diagnostics.csv`
  - `{run_name}_summary.json`
- A smoke run with 40 train and 40 test locality-season groups completed
  successfully. The smoke metrics are not biologically interpretable, but they
  validated the group/checklist joins, full repeated-visit likelihood,
  availability predictions, marginal checklist detection predictions, and output
  writing.
- A later smoke run validated the added latent-diagnostic outputs. These
  outputs deliberately separate headline prior-predictive metrics from
  label-informed diagnostics:
  - `latent_marginal_all_pairs`: fair prior marginal checklist prediction
    before using held-out group detection history
  - `latent_posterior_marginal_all_pairs_label_informed`: diagnostic posterior
    marginal prediction after conditioning on group detections
  - `latent_conditional_detection_known_available_pairs`: diagnostic detection
    component inside locality-season/species groups with at least one detection

Smoke-test command:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 1 --max-groups-per-split 40 --run-name latent_smoke --output-dir data/ebird/locality_season_top100/latent_models_smoke
```

First full latent repeated-visit command:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 20 --run-name latent_repeated_visit_e20
```

First full latent repeated-visit result:

- Retained 450,699 of 661,979 checklists after locality-season filters.
- Latent training data:
  - train groups: 23,617
  - test groups: 9,932
  - train checklists: 315,183
  - test checklists: 127,719
  - species: 100
- Training NLL was still decreasing at epoch 20:
  - epoch 1: 4.37926
  - epoch 10: 4.08252
  - epoch 20: 3.89680
- Checklist-level prior marginal detection metrics:
  - observed detection rate: 0.15359
  - mean predicted detection rate: 0.09903
  - calibration error / ECE: 0.05456
  - BCE: 0.34291
  - micro AUROC / AUPRC: 0.83753 / 0.50844
  - max bin error: 0.14489
- Group-level availability diagnostics:
  - observed positive locality-season/species rate: 0.32871
  - mean predicted availability: 0.37126
  - calibration error versus observed positives: 0.04255
  - positive-triplet AUROC / AUPRC: 0.82679 / 0.70661
  - ECE versus observed positives: 0.04346
  - max bin error versus observed positives: 0.06664
- Interpretation:
  - The first latent model is not yet competitive with the two-component bridge
    as a checklist-level detector.
  - The availability component is promising: it ranks positive
    locality-season/species triplets well, which is closer to the intended
    occupancy-style target.
  - The marginal checklist detector is too conservative. This could be
    optimization underfit, scale mismatch between availability and detection, or
    both.
  - Observed positive locality-season/species rate is a lower bound on true
    availability, so availability calibration against observed positives should
    be treated as a diagnostic rather than true occupancy calibration.
  - Because training NLL was still improving, the next action is a longer run
    using the updated script with posterior and conditional-detection
    diagnostics, not an architecture change.

Next latent repeated-visit command:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 100 --run-name latent_repeated_visit_e100
```

Completed 100-epoch latent repeated-visit run:

- Training NLL continued improving, but with diminishing returns:
  - epoch 20: 3.89680
  - epoch 50: 3.72163
  - epoch 100: 3.65533
- Checklist-level prior marginal detection improved substantially versus e20:
  - mean predicted detection rate: 0.12600 versus observed 0.15359
  - calibration error / ECE: 0.02759
  - BCE: 0.30258
  - micro AUROC / AUPRC: 0.86874 / 0.56294
  - max bin error: 0.10397
- Group-level availability ranking improved:
  - mean predicted availability: 0.43967 versus observed positive rate 0.32871
  - positive-triplet AUROC / AUPRC: 0.84574 / 0.71968
  - calibration error versus observed positives: 0.11097
- Latent detection diagnostics:
  - `latent_marginal_all_pairs`: fair prior predictive metric; micro
    AUROC/AUPRC 0.86874 / 0.56294, BCE 0.30258, ECE 0.02759
  - `latent_posterior_marginal_all_pairs_label_informed`: diagnostic only;
    micro AUROC/AUPRC 0.90664 / 0.59869, BCE 0.25942, ECE 0.01240
  - `latent_conditional_detection_known_available_pairs`: diagnostic only;
    micro AUROC/AUPRC 0.74755 / 0.60026, BCE 0.54975, ECE 0.02708
- Comparison with the current two-component bridge:
  - bridge `two_component`: BCE 0.29104, micro AUROC/AUPRC
    0.87615 / 0.57670, ECE 0.00664
  - latent prior marginal is close on ranking but worse on calibration and BCE
  - latent posterior label-informed diagnostic is stronger than the bridge,
    showing that repeated-visit information can materially improve predictions
    when group availability is inferred from visit history
- Species-level pattern from the latent diagnostic script:
  - largest latent AUPRC gains versus the bridge include White-eyed Vireo,
    Northern Parula, Blue-gray Gnatcatcher, Dark-eyed Junco, Hooded Warbler,
    Golden-crowned Kinglet, Red-eyed Vireo, Ruby-throated Hummingbird, Eastern
    Wood-Pewee, Black-and-white Warbler, and Wood Thrush
  - largest losses include Laughing Gull, Brown Pelican, Red-breasted Nuthatch,
    Boat-tailed Grackle, Tree Swallow, Royal Tern, Hooded Merganser, American
    Herring Gull, Bald Eagle, and Brown-headed Nuthatch
  - the losses are concentrated among coastal/waterbird and some sparse-support
    species, which should be checked before changing the framework globally
- Focus-species season diagnostics show remaining underprediction in high-rate
  seasons, especially Eastern Towhee, Double-crested Cormorant, Northern
  Cardinal, Wood Thrush early breeding, and Green Heron late breeding.
- Interpretation:
  - e100 largely confirms that e20 was underfit.
  - The latent path remains valid and scientifically better aligned with the
    repeated-checklist structure than more checklist-only architecture tuning.
  - The fair prior marginal model still underpredicts detection, so the next
    modeling issue is component-scale/identifiability calibration: the model can
    trade off high availability and low conditional detection while preserving
    much of the likelihood.
  - Do not treat availability calibration versus observed positive groups as
    true calibration. Observed positives are lower bounds on availability.

Latent diagnostic command:

```
python exp/diagnose_ebird_latent_repeated_visit.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_repeated_visit_e100 --bridge-dir data/ebird/locality_season_top100/detection_models --bridge-run-name two_component_checklist_detection_e10_d20 --compare-run latent_repeated_visit_e20
```

Completed 200-epoch latent repeated-visit run:

- Training NLL continued improving slowly:
  - epoch 100: 3.65533
  - epoch 150: 3.63413
  - epoch 200: 3.62422
- Prior marginal checklist detection:
  - mean predicted detection rate: 0.12801 versus observed 0.15359
  - calibration error / ECE: 0.02558
  - BCE: 0.29878
  - micro AUROC / AUPRC: 0.87234 / 0.56802
  - max bin error: 0.09577
- Group-level availability:
  - mean predicted availability: 0.44108 versus observed positive rate 0.32871
  - positive-triplet AUROC / AUPRC: 0.84384 / 0.71556
  - calibration error versus observed positives: 0.11238
- Label-informed posterior diagnostic:
  - mean predicted detection rate: 0.14882 versus observed 0.15359
  - BCE: 0.25749
  - micro AUROC / AUPRC: 0.90808 / 0.60025
  - ECE: 0.01171
- Change from e100:
  - prior marginal micro AUPRC improved by 0.00508
  - prior marginal BCE improved by 0.00379
  - prior marginal calibration error improved by 0.00201
  - availability AUROC/AUPRC declined by 0.00189 / 0.00412
- Interpretation:
  - More epochs help, but the gains after e100 are small.
  - The remaining prior marginal underprediction is unlikely to be solved by
    simply running much longer.
  - The next defensible probe is a training-time moment constraint on the
    marginal detection rate. This tests whether a light identifiability anchor
    can improve prior marginal calibration while preserving the repeated-visit
    availability signal.

Latent e200 diagnostic command:

```
python exp/diagnose_ebird_latent_repeated_visit.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_repeated_visit_e200 --bridge-dir data/ebird/locality_season_top100/detection_models --bridge-run-name two_component_checklist_detection_e10_d20 --compare-run latent_repeated_visit_e100
```

Latent marginal-rate moment probe:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 100 --run-name latent_repeated_visit_e100_mrate100 --marginal-rate-l2 100
```

Completed global marginal-rate moment probe:

- Command:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 100 --run-name latent_repeated_visit_e100_mrate100 --marginal-rate-l2 100
```

- Training:
  - epoch 1 objective/NLL/rate penalty: 4.89653 / 4.37926 / 0.51727
  - epoch 50 objective/NLL/rate penalty: 3.77493 / 3.75150 / 0.02343
  - epoch 100 objective/NLL/rate penalty: 3.67663 / 3.67230 / 0.00433
- Prior marginal checklist detection:
  - mean predicted detection rate: 0.14066 versus observed 0.15359
  - calibration error: 0.01293
  - BCE: 0.30110
  - micro AUROC / AUPRC: 0.86636 / 0.55854
  - ECE: 0.01498
  - max bin error: 0.07787
- Group-level availability:
  - mean predicted availability: 0.46377 versus observed positive rate 0.32871
  - positive-triplet AUROC / AUPRC: 0.84532 / 0.71766
  - calibration error versus observed positives: 0.13506
- Label-informed posterior diagnostic:
  - mean predicted detection rate: 0.15636 versus observed 0.15359
  - BCE: 0.25979
  - micro AUROC / AUPRC: 0.90614 / 0.59802
  - ECE: 0.00583
- Conditional detection in known-available groups:
  - mean predicted detection rate: 0.32508 versus observed 0.33381
  - BCE: 0.54923
  - micro AUROC / AUPRC: 0.74657 / 0.59957
  - ECE: 0.01082
- Change versus unconstrained e200:
  - prior marginal calibration error improved by 0.01265
  - ECE improved by 0.01060
  - max bin error improved by 0.01790
  - BCE worsened by 0.00232
  - micro AUROC/AUPRC declined by 0.00598 / 0.00947
  - availability AUPRC improved slightly by 0.00210, while
    availability-vs-observed-positive calibration worsened
- Species-level pattern:
  - the largest gains versus the bridge remained mostly forest/interior species
    such as White-eyed Vireo, Dark-eyed Junco, Northern Parula,
    Ruby-throated Hummingbird, Red-eyed Vireo, and Golden-crowned Kinglet
  - the largest losses worsened for Laughing Gull, Brown Pelican,
    Boat-tailed Grackle, Royal Tern, Red-breasted Nuthatch,
    Great Black-backed Gull, and American Herring Gull
- Interpretation:
  - The global moment anchor works as a calibration lever, but at this strength
    and training horizon it introduces a ranking/species tradeoff.
  - Because this run used 100 epochs while the current unconstrained reference is
    e200, the next clean test is to run the same anchor for 200 epochs before
    deciding whether the issue is the anchor strength or simply shorter
    optimization.

Latent e100 marginal-rate diagnostic command:

```
python exp/diagnose_ebird_latent_repeated_visit.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_repeated_visit_e100_mrate100 --bridge-dir data/ebird/locality_season_top100/detection_models --bridge-run-name two_component_checklist_detection_e10_d20 --compare-run latent_repeated_visit_e200
```

Completed 200-epoch global marginal-rate moment probe:

- Command:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate100 --marginal-rate-l2 100
```

- Training:
  - epoch 100 objective/NLL/rate penalty: 3.67663 / 3.67230 / 0.00433
  - epoch 150 objective/NLL/rate penalty: 3.65075 / 3.64777 / 0.00298
  - epoch 200 objective/NLL/rate penalty: 3.63836 / 3.63586 / 0.00250
- Prior marginal checklist detection:
  - mean predicted detection rate: 0.14203 versus observed 0.15359
  - calibration error: 0.01156
  - BCE: 0.29745
  - micro AUROC / AUPRC: 0.87111 / 0.56565
  - ECE: 0.01733
  - max bin error: 0.07731
- Group-level availability:
  - mean predicted availability: 0.47396 versus observed positive rate 0.32871
  - positive-triplet AUROC / AUPRC: 0.84647 / 0.71567
  - calibration error versus observed positives: 0.14526
- Label-informed posterior diagnostic:
  - mean predicted detection rate: 0.15545 versus observed 0.15359
  - BCE: 0.25798
  - micro AUROC / AUPRC: 0.90785 / 0.60017
  - ECE: 0.00800
- Conditional detection in known-available groups:
  - mean predicted detection rate: 0.32237 versus observed 0.33381
  - BCE: 0.54446
  - micro AUROC / AUPRC: 0.75109 / 0.60185
  - ECE: 0.01175
- Change versus unconstrained e200:
  - prior marginal calibration error improved by 0.01402
  - ECE improved by 0.00825
  - max bin error improved by 0.01846
  - BCE improved by 0.00133
  - micro AUROC/AUPRC declined by 0.00123 / 0.00237
  - availability AUROC/AUPRC improved by 0.00262 / 0.00012
  - availability-vs-observed-positive calibration worsened, which is expected
    because observed positive groups remain a lower bound on true availability
- Species-level pattern:
  - the strongest gains versus the bridge include Great Black-backed Gull,
    White-eyed Vireo, Dark-eyed Junco, Ruby-throated Hummingbird,
    Golden-crowned Kinglet, Blue-gray Gnatcatcher, Northern Parula, American
    Herring Gull, Eastern Wood-Pewee, Hooded Warbler, Royal Tern, and
    Black-and-white Warbler
  - the largest losses versus the bridge are now Red-breasted Nuthatch,
    Laughing Gull, Tree Swallow, Hooded Merganser, White-throated Sparrow, Bald
    Eagle, Brown Pelican, Mallard, Bufflehead, Swamp Sparrow,
    White-breasted Nuthatch, and Double-crested Cormorant
  - compared with the 100-epoch moment run, the severe coastal/waterbird losses
    are much smaller, so the earlier collapse was partly undertraining rather
    than only the moment anchor itself
- Interpretation:
  - `latent_repeated_visit_e200_mrate100` is the current calibrated latent
    sensitivity run.
  - It is not strictly dominant over unconstrained e200: it improves pooled
    calibration and BCE, while slightly lowering pooled ranking.
  - It is close enough that the next useful experiment is not a new architecture
    but weaker e200 moment anchors (`25`, `50`) to map the calibration/ranking
    Pareto frontier.

Latent e200 marginal-rate diagnostic command:

```
python exp/diagnose_ebird_latent_repeated_visit.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_repeated_visit_e200_mrate100 --bridge-dir data/ebird/locality_season_top100/detection_models --bridge-run-name two_component_checklist_detection_e10_d20 --compare-run latent_repeated_visit_e200
```

Completed weaker 200-epoch marginal-rate moment probe:

- Command:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate25 --marginal-rate-l2 25
```

- Training:
  - epoch 100 objective/NLL/rate penalty: 3.66392 / 3.65964 / 0.00428
  - epoch 150 objective/NLL/rate penalty: 3.64132 / 3.63812 / 0.00320
  - epoch 200 objective/NLL/rate penalty: 3.63056 / 3.62771 / 0.00286
- Prior marginal checklist detection:
  - mean predicted detection rate: 0.13655 versus observed 0.15359
  - calibration error: 0.01704
  - BCE: 0.29737
  - micro AUROC / AUPRC: 0.87181 / 0.56674
  - ECE: 0.01955
  - max bin error: 0.07942
- Group-level availability:
  - mean predicted availability: 0.46440 versus observed positive rate 0.32871
  - positive-triplet AUROC / AUPRC: 0.84583 / 0.71755
  - calibration error versus observed positives: 0.13569
- Label-informed posterior diagnostic:
  - mean predicted detection rate: 0.15204 versus observed 0.15359
  - BCE: 0.25789
  - micro AUROC / AUPRC: 0.90783 / 0.60007
  - ECE: 0.00936
- Conditional detection in known-available groups:
  - mean predicted detection rate: 0.31547 versus observed 0.33381
  - BCE: 0.54479
  - micro AUROC / AUPRC: 0.75132 / 0.60192
  - ECE: 0.01846
- Change versus unconstrained e200:
  - prior marginal calibration error improved by 0.00854
  - ECE improved by 0.00603
  - max bin error improved by 0.01635
  - BCE improved by 0.00141
  - micro AUROC/AUPRC declined by 0.00053 / 0.00127
  - availability AUROC/AUPRC improved by 0.00199 / 0.00199
  - availability-vs-observed-positive calibration worsened, as expected for a
    higher latent availability scale
- Species-level pattern:
  - the strongest gains versus the bridge include Great Black-backed Gull,
    White-eyed Vireo, Royal Tern, Boat-tailed Grackle, American Herring Gull,
    Ruby-throated Hummingbird, Golden-crowned Kinglet, Blue-gray Gnatcatcher,
    Dark-eyed Junco, Hooded Warbler, Northern Parula, Eastern Wood-Pewee, and
    Black-and-white Warbler
  - the largest losses versus the bridge remain Tree Swallow, Red-breasted
    Nuthatch, Hooded Merganser, Mallard, White-throated Sparrow, Bald Eagle,
    Laughing Gull, Swamp Sparrow, Bufflehead, Double-crested Cormorant,
    Eastern Kingbird, Downy Woodpecker, White-breasted Nuthatch, and
    Brown-headed Nuthatch
- Interpretation:
  - `latent_repeated_visit_e200_mrate25` is now the best diagnosed calibrated
    latent sensitivity run.
  - It is not as well calibrated as `mrate100`, but it gives up less pooled
    ranking and has slightly better BCE than `mrate100`.
  - The remaining bridge losses are concentrated in species whose detection may
    depend strongly on movement, patchiness, water/coastal geography, or
    localized effort structure. These species should remain in focus
    diagnostics before the latent model is treated as an ecological replacement
    for the two-component detector.

Completed 200-epoch midpoint marginal-rate probe:

- Command:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate50 --marginal-rate-l2 50
```

- Prior marginal checklist detection:
  - mean predicted detection rate: 0.13952 versus observed 0.15359
  - calibration error: 0.01407
  - BCE: 0.29727
  - micro AUROC / AUPRC: 0.87154 / 0.56630
  - ECE: 0.01843
  - max bin error: 0.07806
- Group-level availability:
  - mean predicted availability: 0.47100 versus observed positive rate 0.32871
  - positive-triplet AUROC / AUPRC: 0.84629 / 0.71683
  - calibration error versus observed positives: 0.14229
- Label-informed posterior diagnostic:
  - mean predicted detection rate: 0.15359 versus observed 0.15359
  - BCE: 0.25797
  - micro AUROC / AUPRC: 0.90782 / 0.60010
  - ECE: 0.00882
- Conditional detection in known-available groups:
  - mean predicted detection rate: 0.31847 versus observed 0.33381
  - BCE: 0.54458
  - micro AUROC / AUPRC: 0.75124 / 0.60190
  - ECE: 0.01554
- Change versus unconstrained e200:
  - prior marginal calibration error improved by 0.01151
  - ECE improved by 0.00715
  - max bin error improved by 0.01771
  - BCE improved by 0.00151
  - micro AUROC/AUPRC declined by 0.00081 / 0.00172
  - availability AUROC/AUPRC improved by 0.00244 / 0.00128
  - availability-vs-observed-positive calibration worsened by 0.02991
- Change versus `mrate25`:
  - mean predicted detection rate increased by 0.00297
  - prior marginal calibration error improved by 0.00297
  - ECE improved by 0.00112
  - max bin error improved by 0.00136
  - BCE improved by 0.00010
  - micro AUROC/AUPRC declined by 0.00027 / 0.00044
  - availability AUPRC declined by 0.00072
  - availability-vs-observed-positive calibration worsened by 0.00660
- Species-level pattern:
  - the strongest gains versus the bridge include Great Black-backed Gull,
    White-eyed Vireo, Royal Tern, Boat-tailed Grackle, American Herring Gull,
    Dark-eyed Junco, Ruby-throated Hummingbird, Golden-crowned Kinglet,
    Blue-gray Gnatcatcher, Northern Parula, Hooded Warbler, Eastern Wood-Pewee,
    and Black-and-white Warbler
  - the largest losses versus the bridge are Red-breasted Nuthatch, Tree
    Swallow, Hooded Merganser, Laughing Gull, White-throated Sparrow, Bald
    Eagle, Mallard, Swamp Sparrow, Bufflehead, White-breasted Nuthatch,
    Double-crested Cormorant, Downy Woodpecker, Eastern Kingbird,
    Brown-headed Nuthatch, and Pine Warbler
  - relative to `mrate25`, `mrate50` improves some focus-season calibration
    pockets, especially Double-crested Cormorant and Eastern Towhee, but it
    worsens some species-level AUPRC losses such as Red-breasted Nuthatch,
    Hooded Merganser, Laughing Gull, and White-throated Sparrow
- Interpretation:
  - `mrate50` is a viable midpoint sensitivity, not a clear replacement for
    `mrate25`.
  - The difference between `mrate25` and `mrate50` is small enough that the
    choice should be framed as a calibration/ranking preference, not a model
    breakthrough.
  - The useful global moment range is now mapped: unconstrained e200 for ranking,
    `mrate25` for a conservative calibrated latent sensitivity, `mrate50` for a
    slightly stronger calibrated sensitivity, and `mrate100` as the stronger
    calibration endpoint.
  - Further scalar marginal-rate tuning is unlikely to answer the main
    scientific question. The next step should inspect whether latent
    availability and conditional detection are biologically plausible across
    seasons, environments, and known problematic species.

Latent e200 `mrate50` diagnostic commands:

```
python exp/diagnose_ebird_latent_repeated_visit.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_repeated_visit_e200_mrate50 --bridge-dir data/ebird/locality_season_top100/detection_models --bridge-run-name two_component_checklist_detection_e10_d20 --compare-run latent_repeated_visit_e200
python exp/diagnose_ebird_latent_repeated_visit.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_repeated_visit_e200_mrate50 --bridge-dir data/ebird/locality_season_top100/detection_models --bridge-run-name two_component_checklist_detection_e10_d20 --compare-run latent_repeated_visit_e200_mrate25 --output-dir data/ebird/locality_season_top100/latent_models/diagnostics/latent_repeated_visit_e200_mrate50_vs_mrate25
```

Completed latent e200 marginal-rate sweep comparison:

- Added `exp/compare_ebird_latent_repeated_visit_runs.py`.
- Command:

```
python exp/compare_ebird_latent_repeated_visit_runs.py --runs latent_repeated_visit_e200 latent_repeated_visit_e200_mrate25 latent_repeated_visit_e200_mrate50 latent_repeated_visit_e200_mrate100 --comparison-name latent_e200_mrate_sweep
```

- Outputs:
  - `data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_mrate_sweep/latent_run_summary.csv`
  - `latent_species_vs_bridge.csv`
  - `latent_species_auprc_delta_pivot.csv`
  - `latent_focus_species_season.csv`
  - `latent_focus_species_availability_season.csv`
  - `latent_availability_species.csv`
  - `latent_run_tradeoffs.png`
  - `species_auprc_delta_boxplot.png`
  - `focus_species_season_error.png`
- Run summary:
  - unconstrained e200: AUPRC 0.56802; BCE 0.29878; calibration error/ECE
    0.02558 / 0.02558; availability AUPRC 0.71556; focus-season weighted abs
    error 0.02623
  - `mrate25`: AUPRC 0.56674; BCE 0.29737; calibration error/ECE
    0.01704 / 0.01955; availability AUPRC 0.71755; focus-season weighted abs
    error 0.01871
  - `mrate50`: AUPRC 0.56630; BCE 0.29727; calibration error/ECE
    0.01407 / 0.01843; availability AUPRC 0.71683; focus-season weighted abs
    error 0.01595
  - `mrate100`: AUPRC 0.56565; BCE 0.29745; calibration error/ECE
    0.01156 / 0.01733; availability AUPRC 0.71567; focus-season weighted abs
    error 0.01408
- Interpretation:
  - global anchoring behaves monotonically for pooled/focus-season calibration
    and mostly monotonically for pooled AUPRC loss
  - `mrate25` is the conservative calibrated sensitivity because it gives most
    of the useful calibration improvement with the smallest ranking cost
  - `mrate50` is a stronger calibrated sensitivity if focus-season probability
    level matters more than the small pooled AUPRC difference
  - `mrate100` remains the stronger endpoint, not the default
  - remaining failures are species-specific enough that another global scalar is
    unlikely to help; the next test should use the existing species-wise
    marginal-rate anchor

Completed weak species-wise marginal-rate anchor probe:

- Command:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate25_srate10 --marginal-rate-l2 25 --species-marginal-rate-l2 10
```

- Prior marginal checklist detection:
  - mean predicted detection rate: 0.13829 versus observed 0.15359
  - calibration error: 0.01530
  - BCE: 0.29724
  - micro AUROC / AUPRC: 0.87164 / 0.56634
  - ECE: 0.01859
  - max bin error: 0.07828
- Group-level availability:
  - mean predicted availability: 0.46750 versus observed positive rate 0.32871
  - positive-triplet AUROC / AUPRC: 0.84641 / 0.71809
  - calibration error versus observed positives: 0.13879
- Change versus `mrate25`:
  - mean predicted detection rate increased by 0.00174
  - prior marginal calibration error improved by 0.00174
  - ECE improved by 0.00096
  - max bin error improved by 0.00114
  - BCE improved by 0.00013
  - micro AUROC/AUPRC declined by 0.00017 / 0.00041
  - availability AUROC/AUPRC improved by 0.00058 / 0.00054
  - availability-vs-observed-positive calibration worsened by 0.00310
- Run-comparison summary after adding `srate10`:
  - unconstrained e200: AUPRC 0.56802; BCE 0.29878; calibration error/ECE
    0.02558 / 0.02558; availability AUPRC 0.71556; focus-season weighted abs
    error 0.02623
  - `mrate25`: AUPRC 0.56674; BCE 0.29737; calibration error/ECE
    0.01704 / 0.01955; availability AUPRC 0.71755; focus-season weighted abs
    error 0.01871
  - `mrate50`: AUPRC 0.56630; BCE 0.29727; calibration error/ECE
    0.01407 / 0.01843; availability AUPRC 0.71683; focus-season weighted abs
    error 0.01595
  - `mrate100`: AUPRC 0.56565; BCE 0.29745; calibration error/ECE
    0.01156 / 0.01733; availability AUPRC 0.71567; focus-season weighted abs
    error 0.01408
  - `mrate25_srate10`: AUPRC 0.56634; BCE 0.29724; calibration error/ECE
    0.01530 / 0.01859; availability AUPRC 0.71809; focus-season weighted abs
    error 0.01655
- Species-level pattern:
  - gains versus the bridge remain concentrated in Great Black-backed Gull,
    White-eyed Vireo, Royal Tern, Boat-tailed Grackle, American Herring Gull,
    Ruby-throated Hummingbird, Golden-crowned Kinglet, Blue-gray Gnatcatcher,
    Hooded Warbler, Dark-eyed Junco, Northern Parula, Eastern Wood-Pewee, and
    Black-and-white Warbler
  - persistent losses remain Red-breasted Nuthatch, Tree Swallow, Hooded
    Merganser, White-throated Sparrow, Bald Eagle, Mallard, Laughing Gull, Swamp
    Sparrow, Double-crested Cormorant, Bufflehead, Downy Woodpecker, Eastern
    Kingbird, White-breasted Nuthatch, Brown-headed Nuthatch, and Pine Warbler
- Interpretation:
  - the weak species-wise anchor is a balanced sensitivity, not a breakthrough
  - it improves probability-level behavior more than `mrate25` while preserving
    almost the same pooled ranking as `mrate50`
  - it does not materially resolve the core species-level losses, so one
    stronger species-wise anchor is justified before closing this axis

Latent e200 `mrate25_srate10` diagnostic and comparison commands:

```
python exp/diagnose_ebird_latent_repeated_visit.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_repeated_visit_e200_mrate25_srate10 --bridge-dir data/ebird/locality_season_top100/detection_models --bridge-run-name two_component_checklist_detection_e10_d20 --compare-run latent_repeated_visit_e200_mrate25
python exp/compare_ebird_latent_repeated_visit_runs.py --runs latent_repeated_visit_e200 latent_repeated_visit_e200_mrate25 latent_repeated_visit_e200_mrate50 latent_repeated_visit_e200_mrate100 latent_repeated_visit_e200_mrate25_srate10 --comparison-name latent_e200_mrate_srate_sweep
```

Completed stronger species-wise marginal-rate anchor probe:

- Command:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate25_srate50 --marginal-rate-l2 25 --species-marginal-rate-l2 50
```

- Prior marginal checklist detection:
  - mean predicted detection rate: 0.14139 versus observed 0.15359
  - calibration error: 0.01220
  - BCE: 0.29732
  - micro AUROC / AUPRC: 0.87117 / 0.56524
  - ECE: 0.01711
  - max bin error: 0.07667
- Group-level availability:
  - mean predicted availability: 0.47136 versus observed positive rate 0.32871
  - positive-triplet AUROC / AUPRC: 0.84672 / 0.71715
  - calibration error versus observed positives: 0.14266
- Change versus `mrate25_srate10`:
  - mean predicted detection rate increased by 0.00310
  - prior marginal calibration error improved by 0.00310
  - ECE improved by 0.00148
  - max bin error improved by 0.00161
  - BCE worsened by 0.00008
  - micro AUROC/AUPRC declined by 0.00046 / 0.00110
  - availability AUPRC declined by 0.00094
  - availability-vs-observed-positive calibration worsened by 0.00386
- Run-comparison summary after adding `srate50`:
  - unconstrained e200: AUPRC 0.56802; BCE 0.29878; calibration error/ECE
    0.02558 / 0.02558; availability AUPRC 0.71556; focus-season weighted abs
    error 0.02623
  - `mrate25`: AUPRC 0.56674; BCE 0.29737; calibration error/ECE
    0.01704 / 0.01955; availability AUPRC 0.71755; focus-season weighted abs
    error 0.01871
  - `mrate50`: AUPRC 0.56630; BCE 0.29727; calibration error/ECE
    0.01407 / 0.01843; availability AUPRC 0.71683; focus-season weighted abs
    error 0.01595
  - `mrate100`: AUPRC 0.56565; BCE 0.29745; calibration error/ECE
    0.01156 / 0.01733; availability AUPRC 0.71567; focus-season weighted abs
    error 0.01408
  - `mrate25_srate10`: AUPRC 0.56634; BCE 0.29724; calibration error/ECE
    0.01530 / 0.01859; availability AUPRC 0.71809; focus-season weighted abs
    error 0.01655
  - `mrate25_srate50`: AUPRC 0.56524; BCE 0.29732; calibration error/ECE
    0.01220 / 0.01711; availability AUPRC 0.71715; focus-season weighted abs
    error 0.01373
- Species-level pattern:
  - the largest gains versus the bridge remain Great Black-backed Gull,
    White-eyed Vireo, Royal Tern, Boat-tailed Grackle, Golden-crowned Kinglet,
    Hooded Warbler, Ruby-throated Hummingbird, Dark-eyed Junco, Blue-gray
    Gnatcatcher, American Herring Gull, Northern Parula, Eastern Wood-Pewee,
    and Black-and-white Warbler
  - persistent losses remain Red-breasted Nuthatch, Tree Swallow, Hooded
    Merganser, Laughing Gull, White-throated Sparrow, Bald Eagle, Mallard,
    Swamp Sparrow, Bufflehead, Double-crested Cormorant, Downy Woodpecker,
    White-breasted Nuthatch, Brown Pelican, Eastern Kingbird, and Brown-headed
    Nuthatch
- Interpretation:
  - `srate50` confirms the species-wise rate-anchor direction is a small
    calibration/focus-season adjustment, not a fix for the repeated species
    losses
  - this is useful evidence: the persistent species losses are likely structural
    or regime-specific rather than just species-wise marginal-rate calibration
  - close scalar rate-penalty tuning for now
  - next step: diagnose the recurring losses by species group, prevalence,
    availability inflation, and underprediction before changing the model

Latent e200 `mrate25_srate50` diagnostic and comparison commands:

```
python exp/diagnose_ebird_latent_repeated_visit.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_repeated_visit_e200_mrate25_srate50 --bridge-dir data/ebird/locality_season_top100/detection_models --bridge-run-name two_component_checklist_detection_e10_d20 --compare-run latent_repeated_visit_e200_mrate25_srate10
python exp/compare_ebird_latent_repeated_visit_runs.py --runs latent_repeated_visit_e200 latent_repeated_visit_e200_mrate25 latent_repeated_visit_e200_mrate50 latent_repeated_visit_e200_mrate100 latent_repeated_visit_e200_mrate25_srate10 latent_repeated_visit_e200_mrate25_srate50 --comparison-name latent_e200_mrate_srate_sweep
```

Completed latent species-pattern diagnostic across the e200 sensitivity set:

- Command:

```
python exp/diagnose_ebird_latent_species_patterns.py --comparison-dir data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_mrate_srate_sweep
```

- Outputs:
  - `data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_mrate_srate_sweep/species_patterns/persistent_species_summary.csv`
  - `data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_mrate_srate_sweep/species_patterns/species_group_summary.csv`
  - `data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_mrate_srate_sweep/species_patterns/worst_persistent_losses.csv`
  - `data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_mrate_srate_sweep/species_patterns/best_persistent_gains.csv`
  - `data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_mrate_srate_sweep/species_patterns/focus_species_season_persistent_errors.csv`
  - `data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_mrate_srate_sweep/species_patterns/species_group_delta_auprc.png`
  - `data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_mrate_srate_sweep/species_patterns/persistent_loss_scatter.png`
- Broad species-group pattern:
  - water/coastal: 22 species; mean AUPRC delta versus bridge -0.00756; 16
    persistent-loss species; mean latent underprediction 0.01587; mean
    availability minus observed-positive rate 0.10636
  - urban/generalist: 7 species; mean delta -0.00746; 5 persistent-loss
    species; mean underprediction 0.03373; availability minus observed-positive
    rate 0.08326
  - open/agricultural: 16 species; mean delta -0.00664; 8 persistent-loss
    species; mean underprediction 0.01802; availability minus observed-positive
    rate 0.12563
  - raptor/scavenger: 7 species; mean delta -0.00645; 4 persistent-loss
    species; mean underprediction 0.01084; availability minus observed-positive
    rate 0.20666
  - forest/woodland: 29 species; mean delta -0.00245; 15 persistent-loss
    species; mean underprediction 0.01392; availability minus observed-positive
    rate 0.14362
  - other: 19 species; mean delta +0.00315; 8 persistent-loss species; mean
    underprediction 0.01277; availability minus observed-positive rate 0.16176
- Worst persistent losses across all six latent sensitivity runs:
  - Red-breasted Nuthatch: mean AUPRC delta -0.05078
  - Tree Swallow: -0.04987
  - Hooded Merganser: -0.04296
  - White-throated Sparrow: -0.03214
  - Laughing Gull: -0.03081
  - Mallard: -0.02983
  - Bald Eagle: -0.02969
  - Swamp Sparrow: -0.02247
  - Bufflehead: -0.02081
  - Double-crested Cormorant: -0.02049
- Best persistent gains across all six latent sensitivity runs:
  - Great Black-backed Gull: mean AUPRC delta +0.04024
  - White-eyed Vireo: +0.02796
  - Royal Tern: +0.02555
  - Boat-tailed Grackle: +0.02269
  - American Herring Gull: +0.02013
  - Golden-crowned Kinglet: +0.01897
  - Ruby-throated Hummingbird: +0.01894
  - Blue-gray Gnatcatcher: +0.01846
  - Dark-eyed Junco: +0.01795
  - Hooded Warbler: +0.01766
- Worst focus-species season errors across latent runs:
  - Eastern Towhee late breeding: mean latent calibration error 0.05359
  - Northern Cardinal fall migration: 0.04969
  - Double-crested Cormorant fall migration: 0.04638
  - Double-crested Cormorant spring migration: 0.04598
  - Northern Cardinal spring migration: 0.04324
  - Double-crested Cormorant winter: 0.04186
  - Double-crested Cormorant late breeding: 0.04166
- Interpretation:
  - the remaining failures are not isolated to one broad species group; water,
    open, urban, raptor, and woodland species all contain persistent losses
  - the latent model is still generally underpredicting prior marginal
    checklist detections, especially in high-detection resident/focus seasons
    and some waterbird seasons
  - predicted availability being higher than the observed-positive rate should
    not be treated as automatically wrong, because observed positives are a
    lower bound on true availability under imperfect detection
  - additional scalar marginal-rate penalties are unlikely to solve this
    pattern
  - the next useful model change is explicit species-season/phenology structure
    in the latent model, ideally with a partial-pooling option so the method
    remains portable rather than hand-tuned to these NC species

Implemented optional species-season latent offsets:

- Updated `exp/ebird_locality_season_latent_model.py` with:
  - `--species-season-mode none`
  - `--species-season-mode availability`
  - `--species-season-mode detection`
  - `--species-season-mode both`
  - `--species-season-l2`
- The new parameters default to off, so earlier commands and saved runs remain
  reproducible.
- A tiny smoke run passed for the default path and for
  `--species-season-mode detection`.
- Interpretation:
  - the detection-side mode is the first full test because the strongest
    persistent diagnostic is underpredicted prior marginal detection within
    species-season pockets
  - availability-side offsets are available, but should be treated as a second
    test because they may absorb true seasonal availability and make the
    availability/detection boundary less interpretable
  - `both` is intentionally available for sensitivity analysis but should not be
    the first promoted model; it is less identifiable

First full species-season latent test command:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p01 --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.01
```

Follow-up diagnostic command after that run:

```
python exp/diagnose_ebird_latent_repeated_visit.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p01 --bridge-dir data/ebird/locality_season_top100/detection_models --bridge-run-name two_component_checklist_detection_e10_d20 --compare-run latent_repeated_visit_e200_mrate25_srate10
```

Completed species-season detection-offset probe:

- Run:
  `latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p01`
- Training:
  - retained 450,699 of 661,979 checklists after locality-season filters
  - train groups/checklists: 23,617 / 315,183
  - test groups/checklists: 9,932 / 127,719
  - epoch 200 train NLL: 3.60695
  - epoch 200 objective: 3.61221
  - epoch 200 rate penalty: 0.00285
- Fair checklist-level prior marginal metrics:
  - observed detection rate: 0.15359
  - mean predicted detection rate: 0.13867
  - calibration error: 0.01492
  - BCE: 0.29630
  - micro AUROC / AUPRC: 0.87270 / 0.56859
  - ECE: 0.01838
  - max bin error: 0.07662
- Group-level availability diagnostics:
  - observed positive rate: 0.32871
  - mean predicted availability: 0.46768
  - positive-triplet AUROC / AUPRC: 0.84545 / 0.71673
  - calibration error versus observed positives: 0.13897
  - this is still not a true occupancy calibration target, because observed
    positives are a lower bound on true availability under imperfect detection
- Diagnostic-only latent outputs:
  - label-informed posterior marginal AUPRC: 0.60346
  - label-informed posterior BCE: 0.25644
  - known-available conditional detection AUPRC: 0.60524
  - known-available conditional detection BCE: 0.54160
- Change versus `latent_repeated_visit_e200_mrate25_srate10`:
  - mean predicted detection rate: +0.00038
  - calibration error: -0.00038
  - BCE: -0.00094
  - micro AUROC: +0.00106
  - micro AUPRC: +0.00225
  - ECE: -0.00021
  - max bin error: -0.00166
  - availability AUROC / AUPRC: -0.00096 / -0.00136
  - availability calibration error versus observed positives: +0.00018
- Comparison summary across latent references:
  - compared runs:
    - `latent_repeated_visit_e200`
    - `latent_repeated_visit_e200_mrate25`
    - `latent_repeated_visit_e200_mrate25_srate10`
    - `latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p01`
  - the species-season detection-offset run has the best prior-marginal AUPRC
    in this comparison set: 0.56859
  - it also has the best BCE: 0.29630
  - it has the lowest prior-marginal calibration error: 0.01492
  - it has the lowest ECE: 0.01838
  - it has the lowest focus-species season weighted absolute error: 0.01604
  - it still does not beat the two-component bridge on fair prior-marginal
    checklist prediction: bridge AUPRC 0.57670, BCE 0.29104, ECE 0.00664
- Species-level gains versus the two-component bridge:
  - Royal Tern: +0.04350 AUPRC
  - Great Black-backed Gull: +0.04132
  - Dark-eyed Junco: +0.04107
  - White-eyed Vireo: +0.03654
  - Brown Pelican: +0.03161
  - Golden-crowned Kinglet: +0.02594
  - Boat-tailed Grackle: +0.02467
  - Hooded Warbler: +0.02246
  - Eastern Wood-Pewee: +0.02020
  - Blue-gray Gnatcatcher: +0.01916
- Species-level losses versus the two-component bridge:
  - Tree Swallow: -0.05218 AUPRC
  - Hooded Merganser: -0.04731
  - Mallard: -0.03218
  - Bald Eagle: -0.03174
  - Red-breasted Nuthatch: -0.02902
  - Bufflehead: -0.02496
  - Double-crested Cormorant: -0.02233
  - White-throated Sparrow: -0.02219
  - Pied-billed Grebe: -0.02112
  - Swamp Sparrow: -0.02048
- Saved-output comparison commands:

```
python exp/compare_ebird_latent_repeated_visit_runs.py --runs latent_repeated_visit_e200 latent_repeated_visit_e200_mrate25 latent_repeated_visit_e200_mrate25_srate10 latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p01 --comparison-name latent_e200_speciesseason_detection_probe
python exp/diagnose_ebird_latent_species_patterns.py --comparison-dir data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_speciesseason_detection_probe
```

- Species-pattern diagnostic after adding the species-season detection offset:
  - broad group means remain mixed or negative versus the bridge:
    - urban/generalist: mean delta -0.00648
    - raptor/scavenger: -0.00619
    - open/agricultural: -0.00565
    - water/coastal: -0.00553
    - forest/woodland: -0.00121
    - other: +0.00420
  - persistent losses remain concentrated in several recurring species:
    Tree Swallow, Hooded Merganser, Red-breasted Nuthatch, Mallard, Bald Eagle,
    White-throated Sparrow, Swamp Sparrow, Bufflehead, Double-crested
    Cormorant, and Downy Woodpecker
  - this means the species-season detection offset is a real improvement, but
    not a full solution to bridge-level species losses
- Interpretation:
  - this is the most encouraging latent-model result so far because it improves
    the calibrated latent reference on several axes at once
  - the improvement is small in pooled terms but directionally important because
    it targets the diagnosed species-season underprediction without post-hoc
    calibration
  - the method remains generalizable: the offset is indexed by species and
    biological season, not by NC-specific geography or hand-selected species
    corrections
  - the next test should bracket the species-season shrinkage strength before
    adding availability-side offsets, GNN structure, or post-hoc calibration

Completed looser species-season detection-offset sensitivity:

- Run:
  `latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025`
- Training:
  - retained 450,699 of 661,979 checklists after locality-season filters
  - train groups/checklists: 23,617 / 315,183
  - test groups/checklists: 9,932 / 127,719
  - epoch 200 train NLL: 3.60660
  - epoch 200 objective: 3.61021
  - epoch 200 rate penalty: 0.00283
- Fair checklist-level prior marginal metrics:
  - observed detection rate: 0.15359
  - mean predicted detection rate: 0.13872
  - calibration error: 0.01488
  - BCE: 0.29629
  - micro AUROC / AUPRC: 0.87269 / 0.56862
  - ECE: 0.01834
  - max bin error: 0.07662
- Group-level availability diagnostics:
  - observed positive rate: 0.32871
  - mean predicted availability: 0.46759
  - positive-triplet AUROC / AUPRC: 0.84534 / 0.71673
  - calibration error versus observed positives: 0.13888
- Change versus the `0.01` species-season detection-offset run:
  - mean predicted detection rate: +0.00005
  - calibration error: -0.00005
  - BCE: -0.00001
  - micro AUROC: -0.00001
  - micro AUPRC: +0.00002
  - ECE: -0.00004
  - max bin error: 0.00000
  - availability AUPRC: 0.00000
  - focus-season weighted error in the full comparison: 0.01596 versus 0.01604
    for `0.01`
- Full comparison summary:
  - `latent_repeated_visit_e200`: AUPRC 0.56802, BCE 0.29878, calibration
    error 0.02558, ECE 0.02558, focus-season weighted error 0.02623
  - `latent_repeated_visit_e200_mrate25`: AUPRC 0.56674, BCE 0.29737,
    calibration error 0.01704, ECE 0.01955, focus-season weighted error 0.01871
  - `latent_repeated_visit_e200_mrate25_srate10`: AUPRC 0.56634, BCE 0.29724,
    calibration error 0.01530, ECE 0.01859, focus-season weighted error 0.01655
  - `speciesseason_detection_l2_0p01`: AUPRC 0.56859, BCE 0.29630,
    calibration error 0.01492, ECE 0.01838, focus-season weighted error 0.01604
  - `speciesseason_detection_l2_0p0025`: AUPRC 0.56862, BCE 0.29629,
    calibration error 0.01488, ECE 0.01834, focus-season weighted error 0.01596
- Persistent species-pattern summary after including `0.0025`:
  - group means versus the two-component bridge remain negative for:
    - urban/generalist: -0.00627
    - raptor/scavenger: -0.00618
    - open/agricultural: -0.00561
    - water/coastal: -0.00528
    - forest/woodland: -0.00098
  - the "other" group remains slightly positive: +0.00474
  - worst persistent latent losses versus the bridge:
    - Tree Swallow: -0.05131 mean AUPRC delta
    - Hooded Merganser: -0.04422
    - Red-breasted Nuthatch: -0.03975
    - Mallard: -0.03202
    - Bald Eagle: -0.03051
    - White-throated Sparrow: -0.02562
    - Bufflehead: -0.02224
    - Swamp Sparrow: -0.02194
    - Double-crested Cormorant: -0.02114
    - Downy Woodpecker: -0.01973
  - best persistent latent gains versus the bridge:
    - Great Black-backed Gull: +0.04197 mean AUPRC delta
    - Royal Tern: +0.03563
    - White-eyed Vireo: +0.03231
    - Dark-eyed Junco: +0.02667
    - Boat-tailed Grackle: +0.02666
    - Golden-crowned Kinglet: +0.02224
- Interpretation:
  - `0.0025` is technically the best saved latent prior-marginal run in the
    current comparison, but only by tiny margins
  - the `0.0025` versus `0.01` difference is not scientifically meaningful by
    itself; both runs support the value of species-season detection structure
  - more detection-side L2 tuning is unlikely to solve the persistent
    species-level bridge losses
  - the next structurally informative test is species-season availability
    structure, because that asks whether the remaining error is partly true
    seasonal availability/phenology rather than checklist-level seasonal
    detection

Completed species-season availability-offset sensitivity:

- Run:
  `latent_repeated_visit_e200_mrate25_srate10_speciesseason_availability_l2_0p01`
- Training:
  - retained 450,699 of 661,979 checklists after locality-season filters
  - train groups/checklists: 23,617 / 315,183
  - test groups/checklists: 9,932 / 127,719
  - epoch 200 train NLL: 3.62816
  - epoch 200 objective: 3.63220
  - epoch 200 rate penalty: 0.00284
- Fair checklist-level prior marginal metrics:
  - observed detection rate: 0.15359
  - mean predicted detection rate: 0.13846
  - calibration error: 0.01513
  - BCE: 0.29698
  - micro AUROC / AUPRC: 0.87167 / 0.56623
  - ECE: 0.01773
  - max bin error: 0.07382
- Group-level availability diagnostics:
  - observed positive rate: 0.32871
  - mean predicted availability: 0.46906
  - positive-triplet AUROC / AUPRC: 0.84588 / 0.71865
  - calibration error versus observed positives: 0.14035
- Change versus the best detection-side species-season run
  (`speciesseason_detection_l2_0p0025`):
  - mean predicted detection rate: -0.00025
  - calibration error: +0.00025
  - BCE: +0.00069
  - micro AUROC: -0.00102
  - micro AUPRC: -0.00239
  - ECE: -0.00061
  - max bin error: -0.00280
  - availability mean predicted availability: +0.00147
  - availability positive-triplet AUROC / AUPRC: +0.00053 / +0.00192
  - availability calibration error versus observed positives: +0.00147
- Full comparison summary:
  - `latent_repeated_visit_e200`: AUPRC 0.56802, BCE 0.29878, calibration
    error 0.02558, ECE 0.02558, focus-season weighted error 0.02623
  - `latent_repeated_visit_e200_mrate25`: AUPRC 0.56674, BCE 0.29737,
    calibration error 0.01704, ECE 0.01955, focus-season weighted error 0.01871
  - `latent_repeated_visit_e200_mrate25_srate10`: AUPRC 0.56634, BCE 0.29724,
    calibration error 0.01530, ECE 0.01859, focus-season weighted error 0.01655
  - `speciesseason_detection_l2_0p0025`: AUPRC 0.56862, BCE 0.29629,
    calibration error 0.01488, ECE 0.01834, focus-season weighted error 0.01596
  - `speciesseason_availability_l2_0p01`: AUPRC 0.56623, BCE 0.29698,
    calibration error 0.01513, ECE 0.01773, focus-season weighted error 0.01708
- Species-level gains versus the two-component bridge:
  - Great Black-backed Gull: +0.04539 AUPRC
  - Royal Tern: +0.03042
  - Boat-tailed Grackle: +0.02936
  - White-eyed Vireo: +0.02817
  - American Herring Gull: +0.02239
  - Golden-crowned Kinglet: +0.01944
  - Ruby-throated Hummingbird: +0.01934
  - Blue-gray Gnatcatcher: +0.01891
  - Hooded Warbler: +0.01881
  - Dark-eyed Junco: +0.01757
- Species-level losses versus the two-component bridge:
  - Tree Swallow: -0.04991 AUPRC
  - Red-breasted Nuthatch: -0.04975
  - Hooded Merganser: -0.04257
  - White-throated Sparrow: -0.03198
  - Bald Eagle: -0.03035
  - Mallard: -0.02986
  - Laughing Gull: -0.02456
  - Swamp Sparrow: -0.02243
  - Downy Woodpecker: -0.02169
  - Double-crested Cormorant: -0.02092
- Persistent species-pattern diagnostic after adding the availability-side run:
  - group means versus the two-component bridge remain negative for:
    - urban/generalist: -0.00672
    - raptor/scavenger: -0.00628
    - water/coastal: -0.00580
    - open/agricultural: -0.00578
    - forest/woodland: -0.00141
  - the "other" group remains slightly positive: +0.00401
  - worst persistent losses remain Tree Swallow, Red-breasted Nuthatch,
    Hooded Merganser, Mallard, Bald Eagle, White-throated Sparrow,
    Swamp Sparrow, Bufflehead, Double-crested Cormorant, and Downy Woodpecker
  - the availability-side run helps some water/coastal species relative to the
    detection-side run but worsens some woodland/seasonal species, especially
    Red-breasted Nuthatch and focus-season errors
- Interpretation:
  - availability-side species-season structure is informative but not preferred
    as a standalone replacement
  - it improves availability ranking and pooled ECE/max-bin error, but the
    project target is not availability ranking alone; fair prior-marginal
    detection and focus-season plausibility matter
  - the detection-side species-season model remains the better current latent
    sensitivity for the main predictive/plausibility balance
  - a `both` species-season sensitivity is now reasonable as a diagnostic, but
    it should be treated cautiously because it gives both components seasonal
    flexibility and can make the availability/detection boundary less
    identifiable

Completed combined availability-and-detection species-season sensitivity:

- Run:
  `latent_repeated_visit_e200_mrate25_srate10_speciesseason_both_l2_0p01`
- Training:
  - retained 450,699 of 661,979 checklists after locality-season filters
  - train groups/checklists: 23,617 / 315,183
  - test groups/checklists: 9,932 / 127,719
  - epoch 200 train NLL: 3.60582
  - epoch 200 objective: 3.61052
  - epoch 200 rate penalty: 0.00250
- Fair checklist-level prior marginal metrics:
  - observed detection rate: 0.15359
  - mean predicted detection rate: 0.13913
  - calibration error: 0.01447
  - BCE: 0.29593
  - micro AUROC / AUPRC: 0.87282 / 0.56864
  - ECE: 0.01753
  - max bin error: 0.07167
- Group-level availability diagnostics:
  - observed positive rate: 0.32871
  - mean predicted availability: 0.47124
  - positive-triplet AUROC / AUPRC: 0.84435 / 0.71589
  - descriptive calibration error versus observed positives: 0.14253
- Change versus the detection-only species-season run at L2 `0.0025`:
  - mean predicted detection rate: +0.00041
  - calibration error: -0.00041
  - BCE: -0.00035
  - micro AUROC: +0.00013
  - micro AUPRC: +0.00002
  - ECE: -0.00080
  - max bin error: -0.00495
  - availability positive-triplet AUROC / AUPRC: -0.00099 / -0.00084
  - availability descriptive calibration error: +0.00365
- Full saved-output comparison:
  - `latent_repeated_visit_e200`: AUPRC 0.56802, BCE 0.29878, calibration
    error 0.02558, ECE 0.02558, availability AUPRC 0.71556, focus-season
    weighted error 0.02623
  - `latent_repeated_visit_e200_mrate25`: AUPRC 0.56674, BCE 0.29737,
    calibration error 0.01704, ECE 0.01955, availability AUPRC 0.71755,
    focus-season weighted error 0.01871
  - `latent_repeated_visit_e200_mrate25_srate10`: AUPRC 0.56634, BCE 0.29724,
    calibration error 0.01530, ECE 0.01859, availability AUPRC 0.71809,
    focus-season weighted error 0.01655
  - detection-only species-season `0.0025`: AUPRC 0.56862, BCE 0.29629,
    calibration error 0.01488, ECE 0.01834, availability AUPRC 0.71673,
    focus-season weighted error 0.01596
  - availability-only species-season `0.01`: AUPRC 0.56623, BCE 0.29698,
    calibration error 0.01513, ECE 0.01773, availability AUPRC 0.71865,
    focus-season weighted error 0.01708
  - both-components species-season `0.01`: AUPRC 0.56864, BCE 0.29593,
    calibration error 0.01447, ECE 0.01753, availability AUPRC 0.71589,
    focus-season weighted error 0.01604
- Parameter-scale comparison:
  - detection-only run: availability-season RMS 0.00000 and detection-season
    RMS 0.55843
  - both-components run: availability-season RMS 0.41067 and detection-season
    RMS 0.52154
  - the extra availability offset therefore absorbed substantial seasonal
    variation without producing a meaningful ranking or species-level gain
- Persistent losses remain broad. The worst six-run mean AUPRC deltas versus
  the bridge are Tree Swallow -0.05111, Hooded Merganser -0.04401,
  Red-breasted Nuthatch -0.04105, Mallard -0.03175, Bald Eagle -0.03059,
  White-throated Sparrow -0.02661, Bufflehead -0.02207, Swamp Sparrow
  -0.02198, Double-crested Cormorant -0.02114, and Downy Woodpecker -0.02023.
- Interpretation and decision:
  - the `both` run is a small pooled calibration/BCE sensitivity, not a new
    preferred structural model
  - it does not clearly beat the detection-only run because AUPRC is tied,
    focus-season error is slightly worse, availability ranking is slightly
    worse, and the target species failures persist
  - allowing both components the same species-season degree of freedom makes
    their boundary less identifiable; the parameter split confirms that the
    two channels can redistribute the same seasonal signal
  - close species-season placement and L2 tuning
  - retain the detection-only `0.0025` run as the more parsimonious latent
    sensitivity, and retain the `both` run only as evidence that its modest
    calibration gain is available at the cost of greater component ambiguity
  - proceed to component-by-replication-support diagnostics before changing
    model form

Completed component-by-replication-support run:

- Run:
  `latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025_components`
- The run exactly reproduced the preferred detection-only headline metrics:
  prior-marginal AUPRC 0.56862, BCE 0.29629, calibration error 0.01488, and
  ECE 0.01834.
- Fair group any-detection calibration by checklist count:
  - 1-3 checklists: +0.03440 signed error
  - 4-5 checklists: +0.03775
  - 6-10 checklists: +0.03914
  - 11+ checklists: -0.01765
- Fair group any-detection calibration by distinct-date count:
  - 1-2 dates: +0.04992
  - 3-4 dates: +0.03623
  - 5-9 dates: +0.03563
  - 10+ dates: -0.01823
- Effort-diversity support is particularly informative:
  - two duration bins: group any-detection error +0.04306
  - three or more duration bins: +0.00704
  - one protocol: +0.02624
  - two or more protocols: +0.01514
- Checklist-level prior marginal detection shows a different pattern. It is
  nearly aligned in low-checklist strata but increasingly underpredicts in
  intensively sampled groups:
  - 1-3 checklists: +0.00290 signed error
  - 6-10 checklists: -0.00347
  - 11+ checklists: -0.02003
  - 10+ dates: -0.02054
  - 6+ unique observers: -0.02368
- Known-positive conditional-detection diagnostics must be interpreted as
  label-informed. Their apparent underprediction is strongest for groups with
  few visits because conditioning on at least one detection mechanically
  selects high realized detection rates. In the highest-support strata, this
  diagnostic is approximately aligned:
  - 11+ checklists: +0.00293 signed error
  - 10+ dates: +0.00245
  - three or more duration bins: -0.00413
  - two or more protocols: +0.00202
- The 6+ unique-observer stratum is a notable exception. Group any-detection is
  underpredicted by 0.05943 and checklist marginal detection by 0.02368. This
  likely combines observer-rich hotspots, locality quality, and observer
  geography; it should be retained as a transfer stress regime rather than
  filtered away.
- Persistent-loss species show the same distinction. Mean absolute group
  any-detection error across the ten main persistent-loss species improves from
  0.05020 with two duration bins to 0.01075 with three or more, while
  checklist-level error rises from 0.01363 to 0.02449. Stronger replication
  therefore helps the group availability/detection decomposition, but it does
  not by itself solve checklist-frequency prediction in intensively sampled
  localities.
- Interpretation and decision:
  - the latent path remains supported; fair group-level predictions become
    more stable with stronger date and effort diversity
  - the evidence does not support simply discarding all low-checklist groups or
    filtering by observer count
  - conditional `p` appears broadly reasonable in the strongest support
    strata, so the current bottleneck is more consistent with unmodeled
    locality/ecological heterogeneity and observer-geography concentration than
    with a global effort-response failure
  - run one stricter-support sensitivity using at least five distinct dates and
    three duration bins; this retains 11,243 training groups, 4,564 test groups,
    250,466 training checklists, and 99,471 test checklists
  - do not compare its pooled AUPRC directly with the full-support run as if
    they shared a target population. Judge whether component calibration and
    species-season behavior stabilize within the better-supported population
    before deciding on the next model change

Completed stricter replication-support sensitivity:

- Run:
  `latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025_strictsupport`
- Support criteria and retained data:
  - minimum five distinct dates
  - minimum three duration bins
  - 11,243 training groups / 4,564 test groups
  - 250,466 training checklists / 99,471 test checklists
- Fair checklist-level prior marginal metrics on the changed strict-support
  population:
  - observed / predicted detection rate: 0.16115 / 0.14754
  - calibration error: 0.01361
  - BCE: 0.30566
  - micro AUROC / AUPRC: 0.87046 / 0.57498
  - ECE: 0.01741
  - max bin error: 0.06245
- Group-level availability diagnostics on the changed population:
  - observed any-positive rate: 0.41331
  - mean predicted availability: 0.50488
  - positive-triplet AUROC / AUPRC: 0.85919 / 0.80630
  - the apparent availability-versus-positive calibration gap is 0.09158, but
    this remains a lower-bound detection comparison rather than true occupancy
    calibration
- Component results:
  - 10+ dates: group any-detection signed error +0.00112
  - 11+ checklists: +0.00175
  - 5-9 dates: +0.05036
  - 6-10 checklists: +0.05200
  - overall strict-support group any-detection error: +0.02334
  - 10+ dates checklist marginal error: -0.01641
  - 11+ checklists checklist marginal error: -0.01597
  - 5-9 dates checklist marginal error: +0.00137
  - 6-10 checklists checklist marginal error: +0.00063
- Observer strata remain asymmetric:
  - one observer: group any-detection +0.07884, checklist marginal -0.00447
  - 3-5 observers: group +0.01359, checklist -0.01613
  - 6+ observers: group -0.03566, checklist -0.01936
- The label-informed known-positive conditional-detection diagnostic improved
  overall from 0.01535 to 0.00969 absolute rate error, but it cannot establish
  fair calibration because conditioning on at least one detection selects on
  the held-out labels.
- The largest species-season checklist underpredictions remain concentrated in
  common or strongly seasonal species, including American Robin winter,
  American Goldfinch across several seasons, Yellow-rumped Warbler winter,
  White-throated Sparrow spring, and Double-crested Cormorant fall.
- Interpretation and decision:
  - this run does not justify making strict support the default because the
    target population changed and medium-support group errors did not improve
    uniformly
  - the strongest support strata show that the repeated-visit structure can
    calibrate group any-detection well
  - the combination of overpredicted any-detection with approximately matched
    checklist frequency in medium-support groups, and matched any-detection
    with underpredicted checklist frequency in high-support groups, is
    consistent with excess positive correlation among repeated detections
  - under the current conditional-independence assumption, changing `psi` or
    `p` tends to move both checklist frequency and at-least-one probability in
    the same direction, so it cannot naturally resolve these opposing errors
  - stop tightening support thresholds for now
  - directly test conditional independence with fair observed-versus-predicted
    pairwise co-detection rates before implementing a detection random effect or
    overdispersion term

Completed pairwise co-detection diagnostic:

- Run:
  `latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025_strictsupport_pairdiag`
- This reproduced the strict-support population:
  - retained 16,124 of 34,328 locality-seasons
  - 11,243 training groups / 4,564 test groups
  - 250,466 training checklists / 99,471 test checklists
- Fair prior-marginal checklist metrics were unchanged from the strict-support
  sensitivity:
  - observed / predicted detection rate: 0.16115 / 0.14754
  - calibration error: 0.01361
  - BCE: 0.30566
  - micro AUROC / AUPRC: 0.87046 / 0.57498
- Pairwise co-detection by support showed consistent underprediction of
  repeated co-detections:
  - overall strict-support weighted pair error: -0.03335
  - checklists 4-5: -0.02117
  - checklists 6-10: -0.01732
  - checklists 11+: -0.03360
  - dates 5-9: -0.01505
  - dates 10+: -0.03370
  - protocols 1: -0.03568
  - protocols 2+: -0.03284
  - observers 1: -0.03464
  - observers 2: -0.03955
  - observers 3-5: -0.04230
  - observers 6+: -0.03233
- Interpretation and decision:
  - the sign is consistent across every support stratum, so this is not only a
    low-support artifact
  - the model is predicting reasonable marginal detection rates in some strata
    but spreading detections too evenly across repeated visits
  - this supports adding a shared locality-season/species detection frailty, so
    repeated detections can be positively correlated conditional on
    availability and effort
  - start with a global frailty scale because it is the simplest transferable
    correction; only move to species-specific frailty if the global version
    improves co-detection calibration but leaves structured species residuals

Implemented detection-frailty option:

- `exp/ebird_locality_season_latent_model.py` now supports:
  - `--detection-frailty-mode none|global|species|hierarchical`
  - `--detection-frailty-init`
  - `--detection-frailty-l2`
  - `--detection-frailty-deviation-l2`
  - `--frailty-quadrature-points`
- The frailty model uses Gauss-Hermite quadrature to integrate a
  logistic-normal shared detection random effect within each
  locality-season/species group.
- Group any-detection probabilities, posterior availability diagnostics, and
  pairwise co-detection predictions are computed under the same frailty model
  rather than the old independent-visit approximation.
- Each new run writes `<run_name>_frailty_species.csv` and records scale
  standard deviation, minimum, quantiles, median, and maximum in its summary.
  In `hierarchical` mode, species deviations are centered to mean zero so the
  shared scale and departures remain identified.

Completed global-frailty run:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025_strictsupport_frailty_global --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode global --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --frailty-quadrature-points 7
```

Global-frailty result:

- learned frailty scale: 1.07439 logit units
- fair checklist AUPRC / BCE: 0.57279 / 0.30519
- calibration error / ECE / max-bin error: 0.00888 / 0.01337 / 0.04768
- overall observed / predicted any-detection rate: 0.41331 / 0.41551
- overall weighted pairwise error: -0.01562, improved from -0.03335
- focus-season weighted absolute error: 0.01311, improved from 0.01498
- species-level absolute pairwise error improved for 90 of 100 species, but
  substantial under- and over-correction remains species-structured

Completed independently regularized species-specific frailty run:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025_strictsupport_frailty_species --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode species --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --frailty-quadrature-points 7
```

Species-specific frailty result:

- overall weighted pairwise error: -0.01566 versus -0.01562 for global
- fair checklist AUPRC / BCE: 0.57249 / 0.30521
- calibration error / ECE / max-bin error: 0.00907 / 0.01323 / 0.04727
- overall observed / predicted any-detection rate: 0.41331 / 0.41656
- focus-season weighted absolute error: 0.01355 versus 0.01311 for global
- learned scale mean / RMS / maximum: 1.03631 / 1.04000 / 1.22293; implied
  scale standard deviation was only 0.08755
- absolute species-level pairwise error improved versus global for 37 of 100
  species; the median species worsened slightly
- interpretation: this independently shrunk parameterization is effectively
  tied with global frailty and is not promoted

Completed hierarchical-frailty run:

- The first attempt used an overlong filename prefix. Training completed, but
  Windows rejected the 263-character
  `_focus_species_availability_season.csv` path before the pairwise,
  per-species-frailty, and summary artifacts were written. The partial files
  are incomplete and must not be used as a finished run.
- The script now validates every output path before loading data or training.
  The rerun uses a shorter artifact name with identical model settings.

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_strict_frailty_hier_dev0p01 --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode hierarchical --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --detection-frailty-deviation-l2 0.01 --frailty-quadrature-points 7
```

Hierarchical-frailty result and decision:

- fair checklist AUPRC / BCE: 0.57195 / 0.30538, both slightly worse than
  global frailty at 0.57279 / 0.30519
- calibration error / ECE / max-bin error: 0.00919 / 0.01307 / 0.04688
- observed / predicted overall any-detection rate: 0.41331 / 0.41703; error
  0.00372 versus 0.00220 for global
- overall weighted pairwise error: -0.01535 versus -0.01562 for global, an
  improvement of only 0.00027
- focus-season weighted absolute error: 0.01415 versus 0.01311 for global
- species frailty scales: mean 1.02404, standard deviation 0.17436, minimum
  0.55795, median 1.00399, and maximum 1.44796
- absolute species-level pairwise error improved for 39 of 100 species; the
  median species worsened
- decision: retain global frailty and close the frailty variance axis

Completed promoted-global availability artifact run:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_strict_frailty_global_availdiag --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode global --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --frailty-quadrature-points 7
```

Completed held-out availability audit:

```
python exp/diagnose_ebird_latent_availability.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_strict_frailty_global_availdiag
```

Availability-audit result:

- the rerun reproduced the promoted global model exactly
- Wood Thrush and Green Heron show low winter availability and clear
  spring/breeding increases; Northern Cardinal is stable and high year-round
- binned environmental observable-response shape correlations average about
  0.856-0.910 across the four retained environmental covariates
- Great Egret late/fall observable predictions are high by about 0.089
- Black-and-white Warbler winter and fall show component ambiguity even where
  combined any-detection remains fairly close
- some high-support zero-detection cases receive prior any-detection above
  0.9, so pooled calibration is insufficient
- decision: continue the latent structural path, keep the global likelihood
  fixed, audit historical locality support, and do not add a GNN yet

Extended historical-support audit command:

```
python exp/diagnose_ebird_latent_availability.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_strict_frailty_global_availdiag
```

Cross-regime summary command:

```
python exp/summarize_ebird_split_regime_validation.py --case "primary10x10|data/ebird/graph_top100_spatial_10x10|spatial_gcn_frozen_access_h64_l2_z64|data/ebird/baselines_10x10|both" --case "coastalstress|data/ebird/graph_top100_spatial_10x10_coastalstress|spatial_gcn_frozen_access_h64_l2_z64|data/ebird/baselines/coastalstress|both" --case "regime_features|data/ebird/graph_top100_spatial_10x10_regime|spatial_gcn_frozen_access_h64_l2_z64|data/ebird/baselines|both-regime"
```

Commands to fill missing per-species comparisons:

```
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial_10x10 --baseline-dir data/ebird/baselines_10x10 --top-species 100 --tabular-model mlp --feature-set both --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/spatial_gnn_spatial_gcn_frozen_access_h64_l2_z64_test_species_metrics.csv --output data/ebird/graph_top100_spatial_10x10/spatial_gnn_baselines/spatial_gcn_frozen_access_h64_l2_z64_graph_vs_tabular_species.csv
python exp/compare_ebird_graph_tabular_species.py --graph-dir data/ebird/graph_top100_spatial_10x10_regime --baseline-dir data/ebird/baselines --top-species 100 --tabular-model mlp --feature-set both-regime --split spatial-stratified --graph-species-metrics data/ebird/graph_top100_spatial_10x10_regime/spatial_gnn_baselines/spatial_gnn_spatial_gcn_frozen_access_h64_l2_z64_test_species_metrics.csv --output data/ebird/graph_top100_spatial_10x10_regime/spatial_gnn_baselines/spatial_gcn_frozen_access_h64_l2_z64_graph_vs_tabular_species.csv
```

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

Completed validation phases:

1. Spatial-stratified blocked validation was added to the tabular baseline.
   Later split diagnostics also added explicit coastal-stress test blocks, which
   are useful for regime-transfer stress testing even when aggregate metrics
   drop.
2. Effort-only, ecology-only, combined, and MLP tabular baselines were compared
   under spatial validation. The combined ecological/temporal/effort MLP is the
   main portable non-graph checklist-level baseline.
3. Calibration outputs by probability bin and effort stratum were added. These
   remain central because the project goal is effort-aware modeling, not only
   ranking.
4. Graph bridge and spatial-GNN models were evaluated against all-pairs
   checklist/species targets and per-species metrics. Sampled-edge link metrics
   are retained as diagnostics but are not the fair target for final comparison.
5. Cross-regime validation now compares the preferred frozen-access spatial GNN
   residual against matched tabular baselines on primary 10x10, coastal-stress,
   and regime-feature splits.

Current validation priorities:

1. Treat the latent repeated-visit model as the active structural model family
   and the two-component detector as its fair checklist-level predictive
   benchmark.
2. Evaluate the observable group-level target
   `psi * (1 - product(1 - p_i))` against whether each species was detected at
   least once in the locality-season. Do not treat any-detection as true
   occupancy or use `psi` versus any-detection as literal occupancy
   calibration.
3. Diagnose component behavior by replication support: checklist count, date
   count, duration-bin diversity, protocol diversity, and observer diversity.
   Conditional-detection metrics inside known-positive groups are
   label-informed diagnostics and must remain labeled as such.
4. Carry focus-species season and environmental-response outputs forward as
   biological plausibility checks, especially Wood Thrush and Green Heron
   seasonality versus resident/generalist species such as Northern Cardinal
   and Eastern Towhee.
5. Keep temporal and cross-regime transfer validation for any promoted model:
   held-out season year, primary spatial holdout, coastal-stress holdout,
   low-observer-density regimes, effort strata, and species-level calibration.
6. Treat the completed five-date/three-duration-bin run as a support
   sensitivity, not the new default population. It confirms that the strongest
   support strata can be well calibrated but does not solve medium-support or
   observer-geography errors uniformly.
7. Treat the pairwise co-detection diagnostic as completed evidence against
   the independent repeated-visit detection assumption. The completed global
   frailty model reduced but did not eliminate co-detection underprediction,
   improved fair checklist calibration, and preserved or improved aggregate
   focus-season error. Independently regularized species scales did not improve
   on the global model, and hierarchical partial pooling produced only a
   negligible pairwise improvement while worsening fair checklist and
   focus-season metrics. Retain global frailty and close this variance axis.
8. Interpret availability-vs-observed-positive metrics only as lower-bound
   diagnostics. Fair group any-detection probabilities and held-out
   ecological/temporal plausibility are the relevant checks because true
   occupancy is unobserved.
9. Treat the first held-out availability audit as a qualified pass. Wood
   Thrush, Green Heron, and Northern Cardinal show sensible broad phenology,
   and coarse environmental-response shapes are generally retained. Do not
   call `psi` validated because Black-and-white Warbler winter availability,
   Great Egret seasonal overprediction, Wood Thrush underprediction, and
   high-confidence zero-detection cases remain unresolved.
10. The selected-failure historical audit is complete. About 40% of the top
    200 cases indicate historical ecological/locality overgeneralization, 41%
    had some prior detection and indicate temporal/observer nonstationarity,
    and 9.5% are unsupported localities. Both major mechanisms occur at
    personal locations and multi-observer hotspots.
11. The all-pair transfer-strata diagnostic is complete. It identifies both a
    portable locality-transfer gap and a separate historical-state gap. Seen
    localities calibrate well only after averaging over strongly opposed
    same-season-history errors; naturally unseen localities have materially
    worse species calibration and macro AUPRC.
12. The controlled `temporal-locality` run is complete. Portable ranking is
    stable, but calibration and pairwise repeated-detection agreement weaken at
    held-out localities. Treat this as a qualified transfer pass, not proof that
    probability surfaces are portable without adaptation.
13. The shared adaptive-history correction is a qualified success. It improves
    both demonstrated history-state errors, checklist ranking/BCE, focus-species
    transfer, and pairwise co-detection while leaving no-history predictions
    directly unmodified. Promote it as the adaptive branch, retain the
    no-history model as the portable branch, and do not add species deviations.
14. Add graph structure only to the availability component after those tests
    define a stable non-graph latent baseline. Keep checklist detection and
    effort explicit, and compare the latent availability GNN directly with
    this same model without graph structure.
15. Defer post-hoc calibration and additional species-season flexibility. Both
    can improve pooled probabilities while obscuring whether availability and
    detection are scientifically separated.
16. The controlled no-history versus adaptive-history availability comparison
    is a qualified pass: dominant high-support errors and all four
    environmental-response summaries improved, broad phenology was retained,
    and a minority of species-season and extreme zero-detection cases worsened.
    Both fitted halves of the paired 2022 temporal-locality replication are now
    complete. History-strata diagnostics confirm the intended mechanism with
    exact no-history fallback, and availability diagnostics preserve or improve
    aggregate environmental response while retaining broad phenology. Proceed
    to paired ecological-region, coastal/inland or mountain/piedmont direction,
    and low-observer-density tests with the model fixed.
17. Use the new `temporal-regime` split for directional transfer tests. Define
    each held-out locality from its pre-test historical median only, exclude
    those localities from all training years, and do not use species outcomes
    to choose the tail. Begin with low distance to coastline, then reuse the
    same machinery for high elevation and low observer support. Report these as
    stress tests alongside, not instead of, the representative temporal and
    controlled-locality evaluations. The portable coastal run confirms a
    material ranking and group-calibration transfer gap while retaining a
    stable frailty estimate. Shared history recovers substantial ranking and
    availability performance at revisited localities, but does not improve the
    fair group any-detection error; report that gain only as adaptive value.
    Because the existing coastline and waterbody bands are coarse proximity
    features rather than a complete coastal-habitat representation, complete
    paired diagnostics and a coastal preprocessing/covariate sensitivity audit
    before making a stronger coastal-transfer claim. Then continue with high
    elevation and low observer support using the fixed likelihood.
18. The paired coastal diagnostics are complete. Shared history improves the
    intended historical states and broad ranking but does not remove portable
    coastal underprediction. Treat this as evidence for missing ecological
    representation rather than evidence for another likelihood or history
    variant. Build a national-core covariate pipeline with optional regional
    modules, pilot it as a controlled NC sensitivity, and preserve access and
    observer-geography predictors in the observation process.

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

Completed implementation phases:

1. Preprocessed the bulk EBD and sampling files into checklist, detection, and
   species tables.
2. Sampled NC raster covariates onto checklist points and corrected the raster
   stack/masking workflow.
3. Built top-species graph-ready datasets with spatial-stratified splits,
   coastal-stress splits, and validation scripts.
4. Trained tabular linear/MLP baselines, graph link baselines, all-species
   bridge models, RBF spatial residuals, and spatial-cell GNN variants.
5. Added calibration, effort-strata, species/block, residual-map,
   regime-support, split-candidate, and cross-regime summary diagnostics.
6. Built the locality-season replication dataset and replication-support
   diagnostics.
7. Added and ran the first locality-season binomial baseline script.
8. Added the first two-component checklist detection bridge script that combines
   aggregate locality-season availability with checklist-level effort/timing
   detection features.
9. Ran the first two-component checklist detection bridge and confirmed that it
   improves held-out checklist-level BCE, micro AUPRC, and mean species AUPRC
   over the one-component availability-only and effort-only baselines.
10. Added and ran the first saved-output diagnostics for the two-component
    bridge, including metric deltas, species AUPRC deltas, calibration-bin
    checks, and focus-species/season error comparisons.
11. Added direct effort/locality stratum diagnostics to the two-component bridge
    training script. These write compact stratum-level metrics and deltas while
    predictions are still in memory.
12. Ran the effort/locality stratum diagnostics. The two-component bridge
    improved micro-AUPRC in every reported large stratum, with remaining
    concerns concentrated in county/regime calibration rather than ranking.
13. Extended the saved-output detection diagnostic script so it also summarizes
    stratum deltas and worst stratum calibration when the stratum CSVs exist.
14. Added the next compact calibration layer to the two-component bridge:
    county-season metrics/deltas, county-season probability-bin reliability,
    and focus-species season probability-bin reliability. These are generated
    inside the training run while predictions are still in memory.
15. Added `--min-bin-pairs` to the saved-output detection diagnostic script.
    Use this before interpreting ranked probability-bin calibration outputs;
    otherwise tiny bins with only a few pairs can dominate the table.
16. Added focus-species monthly phenology and environmental-response diagnostic
    outputs to the two-component training script. These are written while
    predictions are still in memory and avoid saving the full checklist-by-
    species prediction matrix.
17. Reran the stable 10/20 two-component model with the updated script. It
    reproduced the headline metrics (`two_component` BCE 0.2910, micro AUROC
    0.8762, micro AUPRC 0.5767, mean species AUROC 0.8154, mean species AUPRC
    0.3881) and generated 120 monthly phenology rows plus 320 environmental-
    response rows.
18. Reran the saved-output diagnostics with `--min-bin-pairs 100`, removing
    tiny-bin artifacts from the ranked county-season and focus-species tables.
19. Added `exp/plot_ebird_locality_season_plausibility.py` to create phenology
    plots, four-panel species response plots, response-error heatmaps, and
    compact summary CSVs from the new diagnostic outputs.
20. Reviewed the generated plausibility plots and confirmed that the dominant
    issue is probability-level drift rather than reversed ecological response
    shapes.
21. Added optional two-component neutral-point penalties:
    - `--two-component-residual-l2` shrinks species checklist intercepts and
      species-specific effort coefficients toward zero
    - `--two-component-availability-weight-l2` shrinks species availability
      multipliers toward one
    Both default to zero, preserving all prior commands and results.
22. Ran the first combined `0.01 / 0.01` shrinkage model. Calibration improved
    materially with a small pooled ranking loss, but monthly phenology and
    species-level effects were mixed.
23. Added `exp/compare_ebird_locality_season_runs.py` for reproducible pooled,
    species, phenology, and environmental-response comparisons across saved
    model runs.
24. Ran the residual-only `0.01` ablation. It produced nearly the same tradeoff
    as combined shrinkage and showed that residual/effort shrinkage drives most
    of the calibration improvement and ranking loss.
25. Fixed the comparison script's Windows filename-length failure by using
    compact hashed output stems and adding `--comparison-name`.
26. Ran the availability-weight-only `0.01` ablation and the complete four-run
    comparison. Availability shrinkage alone was effectively neutral, confirming
    that residual shrinkage drives both the calibration gain and its ranking and
    phenology tradeoffs.
27. Ran residual-only shrinkage at `0.0025` and compared it with the
    unregularized and `0.01` endpoints. It retained most of the useful
    calibration/response improvement with roughly half the ranking and
    phenology cost.
28. Ran residual-only shrinkage at `0.005`. The complete sequence confirmed a
    smooth Pareto tradeoff and closed scalar L2 tuning, with `0.0025` retained
    as the conservative regularized sensitivity run.
29. Added shared and partially pooled effort-response modes to the
    locality-season checklist model. Partial mode uses a shared effort response
    plus centered species deviations and applies residual shrinkage only to
    species-specific corrections.
30. Ran the first partial-pooling effort-response model at residual L2 `0.0025`
    and compared it against the unregularized two-component run and the
    species-specific residual-only `0.0025` run. It produced nearly the same
    ranking as weak species-specific shrinkage, weaker calibration than that
    shrinkage run, and no clear evidence that partial pooling is the next
    preferred model.
31. Ran the fully shared effort-response ablation at residual L2 `0.0025`.
    It reduced micro AUPRC from 0.57670 to 0.56023 and mean species AUPRC from
    0.38812 to 0.36940 relative to the unregularized two-component run, while
    slightly worsening pooled ECE. This confirms that species-specific
    detection/effort responses are necessary in the current framework.
32. Added `exp/ebird_locality_season_latent_model.py`, the first direct latent
    repeated-visit availability/detection model. A small smoke run passed and
    wrote checklist-level marginal detection plus group-level availability
    diagnostics.
33. Added latent-diagnostic outputs that split fair prior-predictive checklist
    metrics from label-informed posterior and known-available diagnostics.
34. Ran the first full 20-epoch latent repeated-visit model. It is underfit as a
    checklist detector but promising as an availability model, with strong
    group-level positive-triplet ranking and clear marginal detection
    underprediction that motivates a longer run.
35. Ran the 100-epoch latent repeated-visit model. It improved strongly over
    e20 and now sits close to the two-component bridge on checklist ranking,
    but the fair prior marginal probabilities remain undercalibrated.
36. Added `exp/diagnose_ebird_latent_repeated_visit.py` and ran it against e100,
    e20, and the two-component bridge. The diagnostic confirms that the
    label-informed posterior latent prediction is strong, while prior marginal
    prediction still needs component-scale calibration.
37. Ran the 200-epoch latent repeated-visit model. It improved prior marginal
    AUPRC and BCE only modestly over e100, with persistent underprediction and
    a small decline in availability ranking.
38. Added optional latent marginal-rate moment penalties:
    `--marginal-rate-l2` for overall prior marginal detection-rate anchoring and
    `--species-marginal-rate-l2` for species-wise rate anchoring. A smoke run
    passed, and defaults preserve all prior latent results.
39. Ran `latent_repeated_visit_e100_mrate100` and diagnosed it against e200 and
    the two-component bridge. It improved pooled marginal calibration but reduced
    AUPRC/BCE and worsened several species deltas, especially coastal/waterbird
    species.
40. Ran `latent_repeated_visit_e200_mrate100` and diagnosed it against
    unconstrained e200 and the bridge. The 200-epoch moment run retained the
    calibration gain with a much smaller ranking penalty and less severe
    species-level degradation than the 100-epoch moment run.
41. Added and ran `exp/diagnose_ebird_latent_species_patterns.py` against the
    full e200 latent sensitivity set. It writes persistent species summaries,
    broad species-group summaries, best/worst persistent deltas, focus-species
    season persistent errors, and two diagnostic plots.
42. Added optional species-by-season offsets to the latent repeated-visit model
    and smoke-tested both the default path and the detection-offset path. These
    offsets are off by default and can be applied to availability, detection, or
    both components.
43. Ran and diagnosed the first full detection-side species-season offset at
    L2 `0.01`, then the looser L2 `0.0025` sensitivity. Both were positive
    relative to the balanced calibrated latent reference; `0.0025` is the best
    saved latent prior-marginal run by tiny margins, but persistent species
    losses remain.
44. Ran and diagnosed the first availability-side species-season offset at
    L2 `0.01`. It improved availability ranking and pooled ECE/max-bin error,
    but hurt fair prior-marginal AUPRC/BCE and worsened focus-season weighted
    error relative to the best detection-side run. It is not promoted as the
    preferred latent sensitivity.
45. Ran and diagnosed the combined availability-and-detection species-season
    offset at L2 `0.01`. It tied the detection-only run on pooled AUPRC and
    modestly improved BCE/calibration, but slightly worsened focus-season error
    and availability ranking and did not fix persistent species losses. This
    closes the current species-season offset axis.
46. Added compact latent component diagnostics to
    `exp/ebird_locality_season_latent_model.py`. Future runs now write:
    - component metrics by replication-support stratum
    - component metrics by species and biological season
    - component metrics by species and replication-support stratum
    These distinguish fair prior group/checklist predictions from
    known-positive-group conditional-detection diagnostics.
47. Reran the preferred detection-only latent sensitivity with component
    outputs. Group any-detection calibration improves with stronger date and
    duration-bin support, while checklist marginal detection is underpredicted
    in the most intensively sampled groups. High-support known-positive
    conditional detection is approximately aligned, pointing toward
    locality/ecological heterogeneity rather than a global effort-response
    failure.
48. Added optional minimum group-support arguments to the latent model:
    `--min-group-checklists`, `--min-group-dates`,
    `--min-group-duration-bins`, `--min-group-protocols`, and
    `--min-group-observers`. They default to no additional filtering, preserving
    earlier commands and results.
49. Ran the five-date/three-duration-bin strict-support sensitivity. The
    strongest support strata achieved nearly exact group any-detection
    calibration, but medium-support group errors and high-support checklist
    underprediction persisted. Further threshold tightening is not the next
    preferred direction.
50. Added pairwise co-detection diagnostics to the latent model. These compare
    observed ordered detection pairs with model-implied pair probabilities by
    support stratum and species-season, providing a direct fair check of the
    conditional-independence assumption.
51. Ran the pairwise co-detection diagnostic on the strict-support latent run.
    Observed repeated co-detections exceeded the model-implied independent
    prediction in every support stratum, confirming a broad repeated-detection
    dependence problem.
52. Added optional logistic-normal detection frailty to the latent repeated-
    visit model. It integrates a shared locality-season/species detection
    random effect with Gauss-Hermite quadrature and updates any-detection,
    posterior availability, and pairwise co-detection diagnostics to use the
    same model-implied probabilities.
53. Ran the global-frailty strict-support sensitivity. It reduced overall
    weighted pairwise co-detection error from -0.03335 to -0.01562, improved
    absolute species-level pairwise error for 90 of 100 species, improved fair
    checklist calibration and focus-season error, and left pooled ranking
    nearly unchanged. The learned frailty scale was 1.07439 logit units.
54. Confirmed that the remaining pairwise error is species-structured: one
    global scale still under-corrects several waterbirds and common species
    while over-correcting others. This meets the pre-specified condition for a
    regularized species-specific frailty sensitivity.
55. Ran the independently regularized species-frailty sensitivity. It was
    effectively tied with global frailty, improved absolute pairwise error for
    only 37 of 100 species, slightly worsened fair AUPRC and focus-season
    error, and learned a narrow scale distribution. This formulation is not
    promoted.
56. Added hierarchical frailty with a shared global scale plus zero-centered,
    L2-regularized species deviations. Added a per-species frailty-scale CSV
    and scale-distribution summaries so partial-pooling stability can be
    diagnosed directly.
57. The first full hierarchical run completed training but failed during
    output writing because its longest Windows path exceeded `MAX_PATH`.
    Added output-path preflight validation before training and replaced the
    overlong artifact prefix with `latent_strict_frailty_hier_dev0p01`. The
    partial long-prefix files are not a complete run.
58. Completed the shortened hierarchical run. It produced only a negligible
    overall pairwise improvement, worsened fair checklist and focus-season
    metrics, and improved absolute species-level pairwise error for only 39 of
    100 species. Retained global frailty and closed the frailty variance axis.
59. Added compact held-out focus-species group predictions to latent-model
    outputs and added `exp/diagnose_ebird_latent_availability.py` for
    phenology, environmental-response, high-support non-detection, and fair
    observable any-detection diagnostics.
60. Reran the promoted global-frailty model with availability artifacts and
    completed the first held-out audit. Broad Wood Thrush, Green Heron, and
    Northern Cardinal phenology is plausible, and binned environmental
    response shapes are mostly preserved. Great Egret seasonal overprediction,
    Black-and-white Warbler winter component ambiguity, Wood Thrush
    underprediction, and localized high-confidence zero-detection groups make
    this a qualified rather than final pass.
61. Extended the availability diagnostic with:
    - per-species/covariate weighted observable MAE and binned shape
      correlations
    - counts of high-support zero-detection groups above 0.5, 0.8, and 0.9
      predicted any-detection probability
    - locality names/types and prior-year same-locality support for the top
      zero-detection cases
    - immediately preceding support year, detections in that year, latest
      positive year, and a compact history-class summary
    These outputs are designed to separate temporal/observer instability from
    missing ecological or locality predictors before transfer testing.
62. Completed the high-confidence failure-history audit. Of the 200 selected
    cases, 80 had prior same-season support with no historical detection, 76
    had prior same-season detections, 19 had prior locality support but no
    detection in any season, six had detections in other seasons, and 19 had
    no prior locality support. Personal locations contributed 128 cases and
    hotspots 72, so neither observer identity nor missing ecology alone is an
    adequate explanation.
63. Added `exp/diagnose_ebird_latent_transfer_strata.py`. It reuses the saved
    held-out focus-species predictions and reports fair observable metrics by
    seen/unseen locality, recent same-season history, observer diversity,
    locality type, and season. It writes pooled and per-species CSVs without
    retraining the latent model.
64. Completed the all-pair transfer-strata diagnostic. Naturally unseen
    localities had +0.0486 signed any-detection error, macro AUPRC 0.5087, and
    mean absolute species calibration error 0.0647 versus +0.0023, 0.6293, and
    0.0157 at seen localities. Within sampled localities, latest-prior-year
    detections were underpredicted by -0.1855 and never-detected same-season
    histories were overpredicted by +0.1106. Added a controlled
    `temporal-locality` split to the latent model so portable locality transfer
    can be tested without using held-out locality history during fitting.
65. Completed the controlled temporal-locality promoted-likelihood run. Ranking
    transferred with nearly unchanged micro and macro AUROC and nearly
    unchanged species AUPRC lift, while checklist and species calibration,
    group any-detection calibration, and pairwise co-detection weakened. The
    global frailty scale remained stable. Retain this as a qualified portable
    baseline and diagnose historical-state strata on the same held-out
    localities before adding a small shared adaptive-history component.
66. Completed the controlled transfer-strata diagnostic. Pooled focus-species
    any-detection calibration was nearly exact, but latest-prior-year detections
    were underpredicted by -0.1849 and never-detected same-season histories were
    overpredicted by +0.1058. Every focus species had the same error direction
    in both strata, closely reproducing the broader temporal diagnostic.
67. Added an optional shared availability-history correction to
    `exp/ebird_locality_season_latent_model.py`. It uses only earlier-year
    same-season records, support-attenuates three history states, and has no
    intercept so absent history yields zero direct correction. Saved focus
    predictions now retain portable and adapted availability surfaces, and
    `exp/diagnose_ebird_latent_transfer_strata.py` writes a within-run
    portable-versus-adapted comparison. One-epoch controlled-split smoke tests,
    output-path preflight, compilation, and diagnostic generation passed.
68. Completed the seed-37 shared adaptive-history run. Checklist AUROC/AUPRC
    improved to 0.88239/0.60996, BCE to 0.30811, and weighted pairwise error to
    -0.02125. The shared term reduced the two targeted history-state errors for
    nearly every focus species and left the no-history stratum unchanged.
69. Promoted shared history as the adaptive branch with the no-history fit kept
    as the portable branch. Species-specific history deviations are deferred.
    The remaining guardrail is an apples-to-apples availability-plausibility
    comparison, because all-species group any-detection underprediction
    worsened to -0.02281 and several focus-species/high-support failures remain.
70. Completed the controlled no-history versus adaptive-history availability
    comparison. Shared history reduced the largest high-support signed errors,
    improved weighted environmental-response MAE for all four tested
    covariates, and retained the major phenology contrasts. Some small
    species-season errors and extreme zero-detection tails worsened, so the
    result promotes a two-surface framework rather than replacing the portable
    model. The next validation is a paired, fixed-design 2022
    temporal-locality replication.
71. Completed the independently fitted 2022 portable temporal-locality run.
    Prevalence-normalized AUPRC was stable, checklist calibration improved,
    pairwise co-detection error narrowed, and the global frailty scale remained
    unchanged relative to the 2023 portable run. The opposing one-observer and
    6+ observer group errors reproduced. Proceed to the matching fixed
    shared-history run without retuning.
72. Completed the fixed 2022 shared-history run. Checklist ranking and BCE
    improved, pairwise co-detection error narrowed to -0.00137, and frailty
    remained stable. As in 2023, overall fair group any-detection
    underprediction worsened. Run the paired history-strata and availability
    diagnostics before deciding whether the adaptive branch has replicated
    mechanistically rather than only predictively.
73. Completed the paired 2022 history-strata diagnostics. The two targeted
    history-state errors improved substantially, no-prior predictions were
    exactly unchanged, and observer/locality-type errors generally narrowed.
    This is a second-year mechanistic replication of shared history. Promote it
    as the adaptive branch while retaining the independently fitted portable
    branch; run paired availability diagnostics before broader transfer tests.
74. Completed the paired 2022 availability diagnostics. All four aggregate
    environmental-response MAEs improved, eight of ten high-support
    focus-species errors improved, broad phenology was retained, and average
    confidence on high-support zero-detection groups fell for every focus
    species. Several upper-tail extremes worsened, so retain both surfaces and
    proceed to directional ecological-regime transfer without further tuning.
75. Added a generic `temporal-regime` split to
    `exp/ebird_locality_season_latent_model.py`. The split selects an inclusive
    feature tail from established test-year localities using only their
    pre-test historical medians, removes selected localities from every
    training year, and records the threshold, requested and actual fractions,
    locality IDs, and held-out/retained profile summaries. Compilation, CLI,
    low-tail, high-tail, and tied-threshold synthetic checks passed. The first
    fixed-design stress test holds out the 20% of established localities nearest
    the coast by historical median coastline distance.
76. Completed the portable coastal-tail stress test. The split held out 269
    established localities with no training overlap and an actual test-group
    fraction of 0.206. Checklist AUROC/AUPRC fell to 0.8021/0.3626 at 0.1244
    prevalence, availability any-positive AUROC/AUPRC were 0.8071/0.7035 at
    0.3865 prevalence, and fair group any-detection error widened to -0.0499.
    Checklist macro AUPRC fell to 0.2690, species calibration MAE rose to
    0.0497, and availability macro AUPRC fell to 0.5372 relative to 0.4214,
    0.0236, and 0.6465 on the representative locality holdout.
    Coastal specialists dominate the largest negative species-season errors.
    Frailty remained stable and pairwise co-detection remained accurate, so
    proceed to the identical shared-history run rather than changing the
    likelihood or tuning the portable model on this stress set.
77. Completed the identical shared-history coastal-tail run. Relative to the
    portable fit, checklist AUROC/AUPRC improved from 0.8021/0.3626 to
    0.8295/0.3950, checklist macro AUPRC improved from 0.2690 to 0.3076,
    availability pooled AUPRC improved from 0.7035 to 0.7841, and availability
    macro AUPRC improved from 0.5372 to 0.6735. Checklist AUPRC improved for
    93 of 100 species. Fair group any-detection error worsened slightly from
    -0.0499 to -0.0547, so this is promoted only as adaptive value at revisited
    localities; it does not repair portable coastal extrapolation.
78. Audited the coastal preprocessing path. Barrier-island points are present
    and receive low generalized-coastline distances, but the hydrography source
    is coarse, the stack uses EPSG:3857 map distances, and the Waterbody layer
    does not adequately represent ocean/sound/estuarine habitat. Reclassify the
    current evaluation as a generalized-coastline proximity stress test. Keep
    it as the coarse-covariate baseline, audit raw-to-final coastal retention,
    and build a metric-CRS coastal-habitat sensitivity branch before making a
    stronger coastal-transfer claim.
79. Completed the paired coastal transfer-strata diagnostics. Shared history
    improved unseen-locality focus-pair AUPRC from 0.7692 to 0.8263 and macro
    AUPRC from 0.5505 to 0.6825. It reduced the two targeted history-state
    errors and improved ranking across every observer-diversity and locality-
    type stratum. The coastal portable gap remains: the adapted model still
    underpredicted unseen-locality prevalence by 0.0568 and hotspot prevalence
    by 0.0692. Retain independent no-history as the portable surface and shared
    history as adaptive value only.
80. Completed the paired coastal availability audit. Shared history improved
    mean observable-response MAE for all four current environmental covariates
    and substantially reduced errors for Double-crested Cormorant, Green Heron,
    Great Egret, Northern Cardinal, and Eastern Meadowlark. It worsened House
    Sparrow and several smaller errors, and did not reduce the overall extreme
    zero-detection tail. The result supports ecological covariate enrichment,
    not another history or likelihood variant. Adopt a national-core plus
    regional-module feature pipeline using authoritative U.S. land-cover,
    vegetation, terrain, hydrography, climate, wetland, and shoreline sources.

Current decisions and immediate implementation steps:

1. Treat the unregularized two-component bridge as the current checklist-level
   detection benchmark:
   - primary ranking reference: `two_component_checklist_detection_e10_d20`
   - conservative calibration sensitivity:
     `two_component_checklist_detection_shrink_r0p0025_a0`
   - fully shared effort is retired as too restrictive
2. Treat `latent_repeated_visit_e200` as the current unconstrained latent
   repeated-visit baseline:
   - e200 improves over e100, but only modestly
   - the posterior diagnostic is strong
   - fair prior marginal detection remains underpredicted
   - availability ranking is useful but availability-vs-observed-positive
     calibration is not true occupancy calibration
3. Treat `latent_repeated_visit_e200_mrate25` as the best diagnosed
   global-rate-only latent sensitivity run from that completed sweep:
   - prior marginal calibration and BCE improve relative to unconstrained e200
   - pooled AUPRC declines only slightly
   - it preserves more ranking than `mrate100`
   - several species still lose ground versus the two-component bridge
   - the result is a Pareto tradeoff, not a strict replacement
4. Treat `latent_repeated_visit_e200_mrate50` as a viable midpoint sensitivity:
   - prior marginal calibration is stronger than `mrate25`
   - pooled BCE is slightly better than `mrate25`
   - pooled AUPRC is slightly lower than `mrate25`
   - species effects are mixed and do not justify replacing `mrate25`
5. Use latent outputs carefully:
   - `latent_marginal_all_pairs` is the fair prior-predictive checklist metric
   - `latent_posterior_marginal_all_pairs_label_informed` is diagnostic only
     because it conditions on held-out group detections
   - `latent_conditional_detection_known_available_pairs` is diagnostic only
     because it evaluates detection inside groups confirmed to be available
6. Treat the global marginal-rate sweep as closed for now. Do not run another
   scalar `marginal-rate-l2` value unless a later biological plausibility check
   points to a specific need.
7. The completed global-rate comparison carried two calibrated sensitivities
   into the later species-wise tests:
   - conservative calibrated latent sensitivity: `mrate25`
   - stronger calibrated latent sensitivity: `mrate50`
   - neither became the final preferred latent sensitivity after the later
     species-wise and species-season diagnostics
8. Treat `latent_repeated_visit_e200_mrate25_srate10` as a balanced sensitivity
   run:
   - slightly better BCE, calibration, availability AUPRC, and focus-season
     error than `mrate25`
   - slightly lower pooled AUPRC than `mrate25`
   - it motivated the stronger species-wise anchor; that axis is now closed by
     the subsequent `srate50` result
9. Treat `latent_repeated_visit_e200_mrate25_srate50` as a stronger calibrated
   sensitivity run:
   - focus-season error and pooled calibration improve further
   - pooled AUPRC declines more
   - recurring species losses remain
   - therefore scalar rate-penalty tuning is closed for now
10. The saved-output species-pattern diagnostic is complete. It shows that
   recurring latent losses are broad rather than confined to one ecological
   group. Rate-penalty tuning is therefore closed for now.

```
python exp/diagnose_ebird_latent_species_patterns.py --comparison-dir data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_mrate_srate_sweep
```

11. Treat the completed species-season detection-offset run
   (`latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p01`)
   as a positive latent-model sensitivity:
   - it improves `latent_repeated_visit_e200_mrate25_srate10` on AUPRC, BCE,
     calibration error, ECE, max-bin error, and focus-season weighted error
   - it is the first latent variant in this sequence to improve ranking and
     calibration together relative to the balanced calibrated reference
   - it still trails the two-component bridge on fair prior-marginal checklist
     prediction
   - persistent species-level bridge losses remain, especially Tree Swallow,
     Hooded Merganser, Mallard, Bald Eagle, Red-breasted Nuthatch,
     Bufflehead, Double-crested Cormorant, White-throated Sparrow,
     Pied-billed Grebe, and Swamp Sparrow
12. Treat the looser species-season detection-offset run
   (`latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025`)
   as the current best saved latent prior-marginal run by tiny margins:
   - AUPRC 0.56862
   - BCE 0.29629
   - calibration error 0.01488
   - ECE 0.01834
   - focus-season weighted error 0.01596
   - the deltas versus `0.01` are negligible, so the detection-side
     species-season L2 axis is closed for now
13. Treat the availability-side species-season run
   (`latent_repeated_visit_e200_mrate25_srate10_speciesseason_availability_l2_0p01`)
   as an informative but non-preferred sensitivity:
   - it improves availability positive-triplet AUPRC to 0.71865
   - it improves pooled ECE/max-bin error relative to the best detection-side
     run
   - it worsens prior-marginal AUPRC from 0.56862 to 0.56623
   - it worsens BCE from 0.29629 to 0.29698
   - it worsens focus-season weighted error from 0.01596 to 0.01708
   - it does not remove persistent species losses
14. The completed `both` species-season sensitivity is not promoted:
   - AUPRC is effectively tied with the detection-only run: 0.56864 versus
     0.56862
   - BCE, calibration error, ECE, and max-bin error improve modestly
   - focus-season weighted error slightly worsens: 0.01604 versus 0.01596
   - availability AUPRC slightly worsens: 0.71589 versus 0.71673
   - persistent species losses remain
   - parameter magnitudes show seasonal signal being redistributed across
     availability and detection, increasing component ambiguity
15. Reproducibility command for the completed `both` run:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate25_srate10_speciesseason_both_l2_0p01 --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode both --species-season-l2 0.01
```

16. Reproducibility command for its pairwise diagnostic:

```
python exp/diagnose_ebird_latent_repeated_visit.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_repeated_visit_e200_mrate25_srate10_speciesseason_both_l2_0p01 --bridge-dir data/ebird/locality_season_top100/detection_models --bridge-run-name two_component_checklist_detection_e10_d20 --compare-run latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025
```

17. Completed full saved-output comparison and species-pattern diagnostics:

```
python exp/compare_ebird_latent_repeated_visit_runs.py --runs latent_repeated_visit_e200 latent_repeated_visit_e200_mrate25 latent_repeated_visit_e200_mrate25_srate10 latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025 latent_repeated_visit_e200_mrate25_srate10_speciesseason_availability_l2_0p01 latent_repeated_visit_e200_mrate25_srate10_speciesseason_both_l2_0p01 --comparison-name latent_e200_speciesseason_both_probe
python exp/diagnose_ebird_latent_species_patterns.py --comparison-dir data/ebird/locality_season_top100/latent_models/diagnostics/run_comparisons/latent_e200_speciesseason_both_probe
```

18. The species-season offset axis is closed. The parsimonious detection-only
   `0.0025` model was reproduced with the new component diagnostics:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025_components --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025
```

19. The component outputs show:
   - `*_component_support_metrics.csv`
   - `*_component_species_season_metrics.csv`
   - `*_component_species_support_metrics.csv`
   - fair group any-detection calibration improves materially with at least
     three duration bins and generally improves with stronger date support
   - checklist marginal detection remains underpredicted in high-support groups
   - known-positive conditional `p` is approximately aligned in the strongest
     support strata but is label-informed
   - the 6+ observer stratum remains a meaningful observer-geography stress
     regime and should not be filtered away
20. Completed stricter replication-support sensitivity using at least five
   distinct dates and three duration bins:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025_strictsupport --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3
```

21. Strict-support interpretation:
   - strongest-support group any-detection calibration is nearly exact
   - medium-support any-detection remains overpredicted
   - high-support checklist detection remains underpredicted
   - strict filtering is informative but is not promoted as the default
   - opposing group/checklist errors point to residual dependence among
     repeated detections rather than another support-threshold choice
22. Completed pairwise co-detection diagnostic for the strict-support run:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025_strictsupport_pairdiag --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3
```

23. Pairwise co-detection result:
   - every replication-support stratum had negative signed error, meaning
     observed repeated co-detection exceeded the conditionally independent
     prediction
   - overall strict-support weighted pair error was -0.03335
   - this confirms that the next model test should be detection
     overdispersion/frailty rather than stricter filters or more seasonal
     offsets
24. Completed global logistic-normal detection frailty on the same
   strict-support population:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025_strictsupport_frailty_global --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode global --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --frailty-quadrature-points 7
```

25. After the frailty run, compare:
   - fair checklist AUPRC changed from 0.57498 to 0.57279, while calibration
     error improved from 0.01361 to 0.00888
   - overall observed / predicted group any-detection was 0.41331 / 0.41551
   - overall weighted pairwise error improved from -0.03335 to -0.01562
   - focus-season weighted absolute error improved from 0.01498 to 0.01311
   - learned global frailty scale was 1.07439 logit units
   - observed-positive availability metrics remain lower-bound diagnostics,
     not true occupancy calibration metrics
26. Completed the regularized species-specific frailty sensitivity because the
   global model improved absolute pairwise error for 90 of 100 species but
   left clear species-structured under- and over-correction:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_repeated_visit_e200_mrate25_srate10_speciesseason_detection_l2_0p0025_strictsupport_frailty_species --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode species --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --frailty-quadrature-points 7
```

27. Species-specific result:
   - overall pairwise error was -0.01566 versus -0.01562 for global
   - fair checklist AUPRC was 0.57249 versus 0.57279 for global
   - overall any-detection error was 0.00325 versus 0.00220 for global
   - focus-season weighted error was 0.01355 versus 0.01311 for global
   - absolute species pairwise error improved for only 37 of 100 species
   - scales remained narrowly distributed around the global result, so this
     parameterization is closed as a non-promoted sensitivity
28. Completed hierarchical frailty with the same global penalty and a separate
   penalty on zero-centered species deviations:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_strict_frailty_hier_dev0p01 --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode hierarchical --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --detection-frailty-deviation-l2 0.01 --frailty-quadrature-points 7
```

29. Hierarchical result:
   - overall pairwise error changed from -0.01562 to -0.01535
   - fair AUPRC declined from 0.57279 to 0.57195
   - overall any-detection error increased from 0.00220 to 0.00372
   - focus-season error increased from 0.01311 to 0.01415
   - absolute species pairwise error improved for only 39 of 100 species
   - retain global frailty and stop this variance-parameter axis
30. Completed the fixed promoted-global rerun that writes group-level
   availability-validation artifacts:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_strict_frailty_global_availdiag --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode global --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --frailty-quadrature-points 7
```

31. Completed the first latent-availability validation battery:

```
python exp/diagnose_ebird_latent_availability.py --latent-dir data/ebird/locality_season_top100/latent_models --run-name latent_strict_frailty_global_availdiag
```

32. Completed the extended diagnostic against the saved predictions:

```
python exp/diagnose_ebird_latent_availability.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_strict_frailty_global_availdiag
```

33. The history classification shows that the selected failures are split
   between two major mechanisms:
   - prior same-season detections followed by recent zero years indicate
     temporal or observer/reporting instability
   - substantial historical same-season support with no detections indicates
     ecological/locality overgeneralization or missing habitat predictors
   - no prior locality support indicates genuine transfer/extrapolation
34. Completed the fair transfer-strata diagnostic across all saved held-out
   focus-species pairs:

```
python exp/diagnose_ebird_latent_transfer_strata.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_strict_frailty_global_availdiag
```

35. Completed the portable baseline under the controlled temporal-locality
   split. Candidate locality sets were chosen without species outcomes, and
   every held-out locality was absent from training:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_strict_frailty_global_localityxfer_s37 --split-mode temporal-locality --test-locality-fraction 0.2 --locality-split-candidates 100 --split-seed 37 --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode global --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --frailty-quadrature-points 7
```

   Ranking remained stable, while checklist calibration error increased to
   0.01583, fair group any-detection was underpredicted by 0.01442, and weighted
   pairwise co-detection was underpredicted by 0.02477. The stable frailty scale
   and stable prevalence-normalized ranking make this a qualified portability
   pass rather than a reason to retune the likelihood.

36. Completed the same fair transfer-strata diagnostic. The controlled test
   contains only `unseen_locality`, and the same-season history errors closely
   reproduce the broader temporal result:

```
python exp/diagnose_ebird_latent_transfer_strata.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_strict_frailty_global_localityxfer_s37
```

37. Completed the first shared, leakage-safe history-adaptive availability
   model on the identical controlled locality split:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --epochs 200 --run-name latent_xfer_s37_histshared_l2_0p01 --split-mode temporal-locality --test-locality-fraction 0.2 --locality-split-candidates 100 --split-seed 37 --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode global --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --frailty-quadrature-points 7 --availability-history-mode shared --availability-history-l2 0.01 --availability-history-support-scale 20
```

38. Completed the fitted history-adapted versus portable-ablation comparison on
   the same held-out pairs. The diagnostic now writes
   `history_adaptation_strata_summary.csv` and
   `history_adaptation_strata_species.csv` in addition to the standard files:

```
python exp/diagnose_ebird_latent_transfer_strata.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_xfer_s37_histshared_l2_0p01
```

39. Completed the adaptive-model availability plausibility battery:

```
python exp/diagnose_ebird_latent_availability.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_xfer_s37_histshared_l2_0p01
```

40. Completed the same availability diagnostic against the controlled
   no-history checkpoint:

```
python exp/diagnose_ebird_latent_availability.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_strict_frailty_global_localityxfer_s37
```

41. The comparison is a qualified pass. Shared history improved the dominant
   high-support errors and every aggregate environmental-response diagnostic;
   broad phenology remained sensible, although Green Heron, Eastern Meadowlark,
   Double-crested Cormorant, and several upper-tail zero-detection cases were
   mixed. Retain both the portable and adaptive surfaces and do not add
   species-specific history coefficients.
42. Completed the independently estimated portable baseline with 2022 held
   out:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --test-season-year 2022 --epochs 200 --run-name latent_xfer22_s37_nohist --split-mode temporal-locality --test-locality-fraction 0.2 --locality-split-candidates 100 --split-seed 37 --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode global --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --frailty-quadrature-points 7
```

   The controlled split retained 633 test groups from 235 held-out localities.
   Checklist AUROC/AUPRC were 0.85787/0.54669 at prevalence 0.15885, giving
   AUPRC lift 3.4415. Mean-rate error/ECE were 0.00231/0.00486, fair group
   any-detection error was -0.01756, weighted pairwise error was -0.00606, and
   the global frailty scale was 1.07885.
43. Completed the fixed shared-history model on the identical 2022 split:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --test-season-year 2022 --epochs 200 --run-name latent_xfer22_s37_histshared_l2_0p01 --split-mode temporal-locality --test-locality-fraction 0.2 --locality-split-candidates 100 --split-seed 37 --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode global --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --frailty-quadrature-points 7 --availability-history-mode shared --availability-history-l2 0.01 --availability-history-support-scale 20
```

   Checklist AUROC/AUPRC improved to 0.86844/0.55758 and BCE to 0.30391.
   Mean-rate error was 0.00116, ECE 0.00647, fair group any-detection error
   -0.02522, weighted pairwise error -0.00137, and global frailty 1.07660.
44. Completed the transfer diagnostics for both checkpoints:

```
python exp/diagnose_ebird_latent_transfer_strata.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_xfer22_s37_nohist
python exp/diagnose_ebird_latent_transfer_strata.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_xfer22_s37_histshared_l2_0p01
```

   The adaptive within-run comparison improved latest-prior-year error from
   -0.2146 to -0.1375 and never-detected error from +0.0812 to +0.0518.
   No-prior and past-detection/recent-zero predictions were unchanged.
   Independent portable versus adapted focus-species AUPRC improved from
   0.8347 to 0.8675 and macro AUPRC from 0.6523 to 0.7503.

   The paired availability diagnostics are complete:

```
python exp/diagnose_ebird_latent_availability.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_xfer22_s37_nohist
python exp/diagnose_ebird_latent_availability.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_xfer22_s37_histshared_l2_0p01
```

   Shared history improved all four aggregate environmental-response MAEs and
   eight of ten high-support focus-species errors while preserving broad
   phenology. Mean confidence on high-support zero-detection groups fell for
   every focus species, although several maxima increased. This passes the
   second-year availability gate and promotes shared history only as the
   adaptive surface; the independent no-history fit remains the portable
   surface.
45. Completed the first fixed-model directional transfer test on the portable
   no-history model. It held out the low 20% tail of established localities by
   pre-2023 median distance to coastline and excluded those localities from all
   training years:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --test-season-year 2023 --epochs 200 --run-name latent_xfer23_coastal20_nohist --split-mode temporal-regime --test-regime-feature distance_to_coastline_m_median --test-regime-tail low --test-regime-fraction 0.2 --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode global --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --frailty-quadrature-points 7
```

   The inclusive threshold was 4,852 EPSG:3857 map meters, with 269 of 1,343 established
   localities and 20.6% of eligible test groups held out. Checklist
   AUROC/AUPRC were 0.8021/0.3626; AUPRC lift over prevalence was 2.916 versus
   3.405 on the representative controlled-locality holdout. Fair group
   any-detection error was -0.0499, ECE was 0.0255, and maximum bin error was
   0.1424. Checklist macro AUPRC was 0.2690 versus 0.4214, species calibration
   MAE was 0.0497 versus 0.0236, and availability macro AUPRC was 0.5372 versus
   0.6465. Stable frailty and pairwise co-detection isolate the main failure to
   portable coastal-regime prediction, especially for coastal specialists.
46. Completed the same deterministic split with the shared adaptive-history
   checkpoint. This measures the benefit of prior same-season locality history
   for revisited localities; it does not change the portable no-history result:

```
python exp/ebird_locality_season_latent_model.py --dataset-dir data/ebird/locality_season_top100 --processed-dir data/ebird/processed_nc_2020_2023 --test-season-year 2023 --epochs 200 --run-name latent_xfer23_coastal20_histshared_l2_0p01 --split-mode temporal-regime --test-regime-feature distance_to_coastline_m_median --test-regime-tail low --test-regime-fraction 0.2 --marginal-rate-l2 25 --species-marginal-rate-l2 10 --species-season-mode detection --species-season-l2 0.0025 --min-group-dates 5 --min-group-duration-bins 3 --detection-frailty-mode global --detection-frailty-init 0.5 --detection-frailty-l2 0.01 --frailty-quadrature-points 7 --availability-history-mode shared --availability-history-l2 0.01 --availability-history-support-scale 20
```

   Shared history improved checklist AUROC/AUPRC to 0.8295/0.3950 and
   availability any-positive AUROC/AUPRC to 0.8618/0.7841. Macro checklist and
   availability AUPRC improved to 0.3076 and 0.6735. It improved checklist
   AUPRC for 93 of 100 species, but fair group any-detection error widened to
   -0.0547. Promote this only as the adaptive surface for revisited localities;
   the independently fitted no-history model remains the portable result.
47. Completed the same transfer-strata and availability diagnostics on both
   coastal checkpoints:

```
python exp/diagnose_ebird_latent_transfer_strata.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_xfer23_coastal20_nohist
python exp/diagnose_ebird_latent_transfer_strata.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_xfer23_coastal20_histshared_l2_0p01
python exp/diagnose_ebird_latent_availability.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_xfer23_coastal20_nohist
python exp/diagnose_ebird_latent_availability.py --latent-dir data/ebird/locality_season_top100/latent_models --dataset-dir data/ebird/locality_season_top100 --run-name latent_xfer23_coastal20_histshared_l2_0p01
```

   Shared history improved the intended historical states and broad ranking,
   but it did not eliminate portable coastal underprediction. Aggregate
   environmental-response errors improved while several species and extreme
   zero-detection cases worsened. Close this diagnostic step without changing
   the likelihood or adding species-specific history.
48. Started the first versioned national covariate feature pipeline as a
   separate NC pilot. Use `EPSG:5070`, preserve source/version/coverage metadata, and
   derive point plus 250 m/1 km/5 km summaries. Stage sources in blocks:
   Annual NLCD/LANDFIRE, NWI/3DHP, 3DEP/Daymet, then C-CAP/CUSP coastal features.
   Keep roads, population, PAD-US/access, and observer geography in the
   observation-process channel by default. Do not overwrite the current
   processed or locality-season datasets. Planning, generic tiled COG/VRT
   processing, Annual NLCD metadata/registration/validation, and the 244-band
   Annual NLCD derivation are implemented. The production catalog resolves nine
   official archives totaling 10.2 GiB. Official raster header access and the
   real NC derivation/QA subsequently completed; see checkpoint 55 for the AOI
   mask correction still required before model reruns.
49. Rebuild a parallel enriched locality-season dataset and rerun the fixed
   portable and shared-history coastal tests without retuning the likelihood.
   Then run the already-supported high-elevation and low-observer-support
   directional tests on both coarse and enriched covariates. These controlled
   comparisons determine whether the transfer gap is coastal-covariate-specific
   or a broader framework limitation. Only after that should the same feature
   definitions be tested in another state or multi-state region.
50. Interpret availability as a latent ecological quantity, not as calibrated
   occupancy. Require fair predicted-vs-observed any-detection agreement,
   biologically sensible phenology/environmental patterns, and bounded
   high-support non-detection behavior.
51. Do not apply post-hoc probability calibration yet. It could improve numeric
   calibration while obscuring whether availability and detection are
   scientifically separated.
52. Keep the frozen-access spatial GNN as the checklist-level benchmark while
   the locality-season model is developed. Do not add more spatial-GNN
   architecture variants unless a diagnostic points to a specific failure mode.
53. Decide whether to rerun the bridge with 30/30 epochs only after the
   response/phenology diagnostics show the current 10/20 result is stable
   enough to justify the longer run.
54. Revisit House Sparrow and Red-breasted Nuthatch. House Sparrow improved in
   the checklist-level two-component bridge, but Red-breasted Nuthatch remains a
   small negative-delta species.
55. Completed the first official full-NC Annual NLCD build and numerical/mapped
   QA, then rejected immediate promotion based on checklist-location QA. The
   build produced 244 bands, 6,588 COGs, and 2,108.33 MiB in 1,993.3 seconds.
   Although all 661,979 checklists were eligible and support exceeded 99.97% at
   every radius, 164 all-radius failures clustered at valid AOI boundary points:
   every point was inside NC but only 4.39-98.11 m from its boundary. Adopt an
   explicit `all_touched` regional raster mask while retaining exact vector AOI
   membership. The resulting planner/raster/NLCD regression suite passes all 14
   tests in 2.744 seconds. The revised plan keeps 27 tiles and adds only 5,295
   boundary cells (`+0.2374%`, 2,235,542 total). The corrected overwrite then
   completed with 244 bands, 6,588 COGs, 2,112.22 MiB, and 2,227.7 seconds;
   its summary records `all_touched`. Corrected numerical/mapped QA passes
   with maximum class-fraction sum error `9.54e-07`, all automated checks
   true, and seven coherent representative previews. Post-rebuild checklist QA
   supports 661,978/661,979 events at 250 m and 1 km and 661,972/661,979 at
   5 km. All-radius failures fall from 164 to one; all seven remaining events
   are marine or marine-likely traveling checklists plotted 3.49-6.66 km
   seaward. Promote Annual NLCD while retaining those values as declared
   terrestrial missingness. Proceed to LANDFIRE without changing the frozen
   locality-season model.
56. Implemented and validated the release-aware LANDFIRE catalog adapter.
   The current official LFPS inventory resolves 13 required CONUS layers: EVT,
   EVC, and EVH for LF2016/LF2022/LF2023 plus exact annual Dist20-Dist23.
   Every selected public ImageServer validates as a one-band 30 m thematic
   raster in `EPSG:5070`. The non-future vegetation mapping is LF2016 for
   2020-2021, LF2022 for 2022, and LF2023 for 2023; corresponding source ages
   are 4, 5, 0, and 0 years. LF2020 vegetation remains an explicit archived
   source gap rather than being backfilled with future LF2022 conditions. The
   revised plan estimates 2,020 logical bands, including 153 LANDFIRE bands.
   This is a metadata/provenance gate only: resolve official release-specific
   class tables and a portable physiognomy/lifeform crosswalk before the
   bounded raster pilot. Keep the latent model frozen during this work.
57. Completed the LANDFIRE class-semantics and bounded interior/coastal
   raw-raster gates.
   The official ImageServer raster-attribute-table endpoint yielded 3,627
   release-specific rows across LF2016/LF2022/LF2023 EVT, EVC, and EVH. Build
   model inputs from a validated nine-class EVT hierarchy
   (`forest_tree`, `shrub`, `herbaceous`, `riparian`, `agriculture`,
   `developed`, `sparse_barren`, `open_water`, and `snow_ice`) while
   preserving every raw release/code/name field. Interpret EVC and EVH as
   cover/height conditional on the mapped dominant lifeform, not simultaneous
   vertical-stratum measurements. A 5 km-buffered LF2023 pilot on
   `xp0014_yp0015` produced three `3668 x 3668` one-band 30 m rasters in
   `EPSG:5070`, each with 100% source coverage and no values outside its exact
   release lookup; total size is 78.86 MiB. The matching
   `xp0017_yp0014` coastal pilot also passed with complete coverage and only
   registered values for all three products. Across both pilots the six files
   total 157.72 MiB. This shows that the earlier marine NLCD gaps are
   source-specific rather than a common AOI/grid failure. Proceed to
   model-scale 250 m/1 km/5 km derivation on both pilots. This enriches the
   ecological availability covariates only; keep the accepted latent
   occupancy-detection model and its observer/process assumptions frozen until
   the covariate rebuild is complete.
58. Completed the bounded LF2023 model-scale vegetation and QA gate. The
   release schema contains 46 logical bands: 27 nine-class EVT fractions,
   nine dominant-lifeform-conditional EVC cover bands, nine corresponding EVH
   height bands, and one modeled-source coverage band. The interior tile
   `xp0014_yp0015` writes 44 COGs and retains two all-NoData 5 km shrub bands
   as empty logical inventories; the coastal tile `xp0017_yp0014` writes all
   46 COGs. Both pass grid, range, VRT order, sparse-band, and class-closure QA
   with maximum EVT fraction-sum error `7.15e-07`. Interior support is 100% at
   all radii. Coastal support is 99.04%, 97.81%, and 90.43% at 250 m, 1 km,
   and 5 km because the modeled-class mask excludes official `Fill-NoData`
   ocean cells. Preserve that support decline rather than imputing vegetation.
   The full covariate regression gate now passes all 35 tests. Proceed to
   Dist20-Dist23 and bounded LF2016/LF2022 checks before statewide LANDFIRE
   materialization; keep the accepted latent model frozen.
59. Completed annual LANDFIRE disturbance semantics/derivation and bounded
   LF2016/LF2022 replication. Dist20-Dist23 now produces 12 exact annual
   disturbance-fraction bands across 250 m, 1 km, and 5 km neighborhoods.
   Official Background is valid undisturbed support; Fill and Water masks are
   excluded from both event numerator and terrestrial denominator. Both
   interior and coastal pilots pass automated and mapped QA. Coastal support
   remains 99.04%, 97.81%, and 90.43% across the three radii, showing the same
   declared terrestrial truncation as vegetation rather than ocean-zero
   imputation. LF2016 and LF2022 each retain 46 nonempty vegetation bands on
   both pilots, with maximum EVT closure error no larger than `8.34e-07`;
   release-to-release maps are coherent and seam-free. A Windows path-length
   failure found during LF2016 was fixed in the shared raster engine by
   shortening only physical artifact names while retaining complete logical
   feature IDs and metadata. The full covariate suite now passes 42 tests.
   Treat vegetation source age as a date-aware extraction/provenance scalar,
   not a spatial raster. The next data step is resumable full-NC LANDFIRE
   orchestration and explicit core/sensitivity materialization profiles.
   Continue to hold the latent likelihood and model parameters fixed.
60. Implemented and validated resumable full-NC LANDFIRE orchestration without
   starting statewide raster execution. The fixed 27-tile plan resolves to 108
   atomic units: 81 tile-by-release vegetation units and 27 tile-level
   Dist20-Dist23 units. A unit is reusable only when its summary, passing QA,
   VRT, inventories, and every referenced COG are present. The manifest hashes
   the grid, source semantics, derivation settings, unit schedule, and profile
   contract; incompatible reuse fails closed. Three shared-COG VRT profiles
   are now explicit: `core` (150 raster bands), `no-structure` (96),
   and `no-disturbance` (138). Release identity and vegetation source age
   remain date-aware row provenance rather than redundant raster bands. The
   real NC dry run reported `0/108` units on its first invocation and
   resumed the same manifest on its second; no source rasters were requested.
   Unit ordering, schema counts, artifact completeness, contract locking, and
   failed-attempt batch limits are covered, and the full covariate suite passes
   50 tests. Begin with a six-unit batch and no `--overwrite`, then inspect
   runtime, storage, and validation before scaling. This remains covariate
   infrastructure work: do not change or refit the accepted locality-season
   likelihood until LANDFIRE and the remaining enriched sources are assembled,
   sampled, and evaluated under the frozen comparison design.
61. Completed the first bounded statewide LANDFIRE production batch. Six
   LF2016 tile/release units advanced the resumable manifest to 6/108 with zero
   failures, one attempt per unit, and no accidental reuse. End-to-end time was
   336 seconds. The batch retained 473.16 MiB of raw official ImageServer
   exports and wrote 34.87 MiB across 268 derived COGs; the complete state-build
   directory is 508.67 MiB. Every unit passes grid, range, COG, VRT-order,
   support, and EVT closure checks, with maximum closure error `8.34e-07`.
   The eight empty logical bands are conditional shrub cover/height features
   in low-support tile/radius combinations and remain represented by empty
   inventories as designed. Support is complete on four units,
   99.9922%/99.9792%/99.8257% on `xp0015_yp0013`, and
   98.2917%/96.9308%/89.5450% on the eastern-edge
   `xp0016_yp0013` at 250 m/1 km/5 km. Keep the next batch at six units:
   it ends on the established coastal gate tile `xp0017_yp0014` and
   provides the next support/resume check before increasing throughput. This
   successful infrastructure batch does not change the frozen
   locality-season model or its promotion criteria.
62. Completed the second bounded statewide LANDFIRE production batch and
   verified real resume behavior. The prior six units were reused without new
   attempts, six new LF2016 units completed on their first attempt, and the
   manifest advanced to 12/108 with zero failures. The new work took 273
   seconds, retained 473.22 MiB of raw exports, and produced 99.23 MiB of
   derived COGs. Cumulative state is 1,081.64 MiB across 1,179 files, including
   542 COGs. Every new unit passes grid, range, COG, VRT-order, support, and
   EVT closure checks; maximum closure error is `8.94e-07`. The coastal gate
   `xp0017_yp0014` retained 99.0350%/97.8138%/90.4327% AOI support at
   250 m/1 km/5 km, consistent with the accepted shoreline/source-edge
   interpretation. Increase the next batch to 15 units to finish all 27
   LF2016 tiles, then stop for release-wide QA before LF2022. Continue without
   `--overwrite`, and keep the accepted locality-season model frozen.
63. Completed and promoted the full 27-tile LF2016 North Carolina release.
   The third production invocation reused the prior 12 units, completed the
   remaining 15 on their first attempts, and advanced the manifest to 27/108
   with zero failures. The new work took 733 seconds, retained 1,183.02 MiB of
   raw exports, and wrote 189.61 MiB across 684 COGs. Cumulative LF2016 state
   is 2,455.58 MiB across 2,658 files, including 2,129.44 MiB raw and 323.70
   MiB across 1,226 COGs. All 27 validations pass; maximum EVT closure error is
   `9.54e-07`. Representative western-edge, interior, coastal, and outer-coast
   previews are coherent and seam-free. A new release-level
   `validate-landfire-checklist-support` gate confirms support for all 661,979
   checklists at 250 m, all but one at 1 km, and all but six at 5 km. The six
   unsupported-at-any-radius records are explicitly pelagic/nearshore
   observations, so terrestrial NoData is correct rather than a coverage
   failure. The full covariate suite passes 52 tests. LF2016 is promoted, but
   the whole LANDFIRE block is not: proceed with one 27-unit LF2022 batch to
   stop at 54/108, repeat release-wide mapped/checklist QA, continue without
   `--overwrite`, and keep the accepted locality-season model frozen.
64. Completed and promoted the full 27-tile LF2022 North Carolina release.
   The production invocation reused all 27 LF2016 units, completed all 27
   LF2022 units on their first attempts, and advanced the manifest to 54/108
   with zero failures. LF2022 took 1,176 seconds, retained 2,129.44 MiB of raw
   exports, and wrote 334.08 MiB across 1,224 COGs. All 27 validations pass;
   maximum EVT closure error is `1.07e-06`. Cumulative LF2016 plus LF2022 state
   is 4,920.26 MiB across 5,319 files and 2,450 COGs. Representative western,
   interior, coastal, and outer-coast maps are coherent and seam-free.
   Checklist support is identical to LF2016: 100.0000%, 99.9998%, and 99.9991%
   at 250 m, 1 km, and 5 km, with the same six explicitly pelagic/nearshore
   unsupported records. Added a reusable `compare-landfire-releases` gate.
   It confirms matching 46-band schemas, zero differences across all 81
   tile/radius support comparisons, exact source-coverage agreement, and
   spatially aligned changes on four representative tiles. Release differences
   must not be interpreted as pure ecological change: they combine landscape
   change with source/classification updates, so release identity and source
   age remain required provenance and `no-structure` remains a sensitivity.
   The complete covariate suite passes 53 tests. Proceed with all 27 LF2023
   units to stop at 81/108, repeat the same release gates, continue without
   `--overwrite`, and keep the accepted locality-season model frozen.
