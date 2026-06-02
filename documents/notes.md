# Notes

For a single species with tabular environmental covariates and a standard presence-only point-process likelihood, a neural IPPP is rarely conceptually different from a flexible IPPP smoother. Neural methods become more compelling when they learn representations from raw spatial data, share information across many species, model sampling bias, or operate at scales where hand-specified smoothers and covariate interactions become impractical.

If the neural model is just

\[
\lambda(s,t,z)=\exp(f_\theta(s,t,z))
\]

fit with the same IPPP likelihood over the same covariates, then it is mostly “a flexible IPPP smoother.” A GAM, spline, or kernel/basis expansion can often play the same role.

Where neural methods become more defensible is when they change something beyond “replace smooth function with MLP.”

**Possible Improvements of a Neural Network over a traditional SDM:**

1. Many-Species Joint Models

Instead of fitting one IPPP per species, use one neural model for many species:

\[
\lambda_j(s,z)=\exp(f_\theta(s,z,e_j))
\]

where \(e_j\) is a learned species embedding.

This can help rare species borrow strength from common species, learn shared environmental responses, and model species-location associations jointly. A recent example is heterogeneous graph neural networks for presence-only SDM, where species and locations are nodes and detections are edges. That is a more meaningful neural use case than a single-species MLP intensity model. See: [Heterogeneous graph neural networks for species distribution modeling](https://arxiv.org/abs/2503.11900).

2. Raw Raster Or Patch Inputs

A traditional IPPP usually receives handcrafted covariates: elevation, canopy, distance to water, climate summaries, etc.

A neural model can ingest image-like patches:

\[
\text{satellite patch around } s \rightarrow f_\theta(s)
\]

This is useful if the signal is in texture, landscape configuration, fragmentation, edge structure, or spatial context that is not captured by scalar raster values. A GAM cannot naturally see “this location is in a riparian corridor surrounded by fragmented forest” unless you manually engineer those features.

This is probably one of the strongest practical arguments for neural SDMs.

3. Spatial Representation Learning Across Huge Domains

Some recent neural SDM work learns spatial embeddings or implicit coordinate representations over large regions, especially with presence-only community-sourced data. The neural gain is not that it is “more nonlinear than an IPPP”; it is that the model learns reusable spatial structure over many locations/species/tasks. See: [Hybrid Spatial Representations for Species Distribution Modeling](https://arxiv.org/abs/2410.10937).

4. Bias / Effort Modeling As A Joint Neural Process

For GBIF/eBird, the biggest problem is often not the ecological intensity. It is the observation process.

Observed records are closer to:

\[
\lambda_{\text{obs}}(s,t)
=
\lambda_{\text{species}}(s,t)
\cdot
\lambda_{\text{observer}}(s,t)
\cdot
p_{\text{detect}}(s,t)
\]

A neural model might help if it jointly learns observer/reporting intensity from many species, checklists, roads, cities, time, platform behavior, or user metadata. But this is no longer just “NIPPP vs IPPP.” It is a richer observation model.

5. Massive-Scale Prediction Pipelines

If you want one model that handles thousands of species, global rasters, temporal covariates, remote sensing, taxonomic embeddings, and transfer to new regions, neural methods become operationally attractive. Not because the point process theory changed, but because the function approximation and representation learning problem got much bigger.

**The Catch:**

Even without absence or pseudo-absence points, you still need some notion of availability. For an IPPP, that is the integral:

\[
\int_W \lambda(u)\,du
\]

In practice, you approximate it with quadrature or Monte Carlo samples over the study window. These are not biological absences, but they are still background/integration points. A neural model also needs this normalization somehow. Pure presences alone do not identify a spatial distribution unless you define the domain/exposure/background measure.

## Bias / Effort Modeling As A Joint Neural Process

This is probably one of the more realistic places where neural methods could
improve GBIF/eBird-style species distribution modeling. The reason is not that
a neural network is inherently a better smoother for one species. The reason is
that presence-only records are samples from an observation process, not direct
samples from the species distribution.

A simple ecological IPPP treats observed records as if they came from the
species intensity:

\[
\lambda_{\text{obs}}(s,t)=\lambda_{\text{species}}(s,t)
\]

For community-science data, a more realistic conceptual model is:

\[
\lambda_{\text{obs},j}(s,t)
=
\lambda_{\text{species},j}(s,t)
\cdot
q_{\text{effort}}(s,t)
\cdot
p_{\text{detect},j}(s,t)
\]

where \(j\) indexes species, \(\lambda_{\text{species},j}\) is the ecological
intensity, \(q_{\text{effort}}\) is observer effort or reporting intensity, and
\(p_{\text{detect},j}\) is detection/reporting probability conditional on
effort.

A standard single-species IPPP fit to GBIF/eBird presences mostly learns
\(\lambda_{\text{obs}}\), not necessarily \(\lambda_{\text{species}}\). That
means it can learn roads, cities, birding hotspots, parks, weekends, platform
behavior, and observer preferences as if they were habitat.

**Why Joint Modeling Helps**

If one species is modeled alone, it is hard to tell whether records cluster near
cities because the species prefers urbanized habitat or because observers are
there. But if many species are modeled together, broad patterns shared across
unrelated species can reveal observation effort.

For example, if many unrelated species have records concentrated near Raleigh,
Charlotte, Asheville, major roads, popular parks, and accessible mountain
trails, that shared pattern is likely to contain a large effort component.
Wood Thrush-specific deviations from that shared pattern are more likely to
represent ecological structure.

Instead of fitting one model per species,

\[
\lambda_j(s,t,z)=\exp(f_j(s,t,z)),
\]

a joint observation model could use something like:

\[
\lambda_{\text{obs},j}(s,t,z,r)
=
\exp\left(
f_\theta(j,s,t,z)
+
g_\phi(s,t,r)
+
h_\psi(j,s,t,o)
\right)
\]

where \(z\) contains environmental covariates, \(r\) contains effort proxies
such as roads, population density, protected areas, checklist density, or
platform usage, and \(o\) contains observer or checklist covariates when
available. The term \(g_\phi\) is a shared effort surface, \(f_\theta\) is the
species-specific ecological component, and \(h_\psi\) can represent
species-specific detectability or reporting behavior.

This is a real modeling extension beyond a neural IPPP of the form:

\[
\lambda_{\text{obs}}(s,t,z)=\exp(f_\theta(s,t,z)).
\]

The neural model is not just replacing a spline with an MLP. It is being used as
part of a multi-task observation model that tries to separate ecology, effort,
and detectability by borrowing information across many species, locations,
times, and observers.

**Why Neural Networks Might Help**

A classical model can include effort terms too. Neural methods become more
appealing when the effort and detection processes are high-dimensional and
messy. Observer effort may depend on distance to roads and trails, population
density, protected areas, season, day of week, time of day, weather, platform
usage over years, observer identity, checklist duration, number of observers,
number of species reported, and spatial reputation of birding hotspots.

A neural model can learn shared latent structure such as:

\[
\text{location embedding}
+
\text{time embedding}
+
\text{observer embedding}
+
\text{species embedding}.
\]

That kind of structure is much harder to specify cleanly in a single-species
GAM or spline IPPP. The strongest argument for a neural approach is therefore
not that it is more nonlinear, but that it can share information across many
related prediction tasks while representing complicated observation bias.

**Useful Data**

For eBird, especially useful fields would include complete checklists,
checklist duration, distance traveled, number of observers, protocol type, time
of day, date, observer identity or an anonymized observer effect, and all
species reported on the checklist. Complete checklists are particularly useful
because they provide information about survey effort and non-detection of
species that were plausibly searched for.

For GBIF, effort metadata are usually weaker. Possible proxies include records
of many taxa as an effort surface, distance to roads or cities, collection
institution, data source, year, coordinate uncertainty, and sampling campaign
metadata when available.

**Failure Mode: Ecology And Effort Can Be Confounded**

This approach can perform poorly when a species is genuinely concentrated in
the same places where observers concentrate. If a species really prefers urban
parks, roadsides, coastal access points, popular wetlands, mountain trails, or
managed preserves, then true habitat and observer effort are aligned.

In that case, a joint effort model may over-correct. It may treat some true
ecological signal as shared observation effort, producing an ecological
intensity map that is too diffuse or that underestimates suitable habitat in
the places where the species actually occurs.

The identifiability problem is:

\[
\text{many records in parks}
\]

could mean:

\[
\text{species likes parks},
\]

\[
\text{observers like parks},
\]

or both.

A joint model can reduce this ambiguity, but it cannot make it disappear unless
the data contain some contrast that separates habitat from effort.

Useful contrast includes complete checklists, repeated visits to the same
places under different effort levels, effort metadata, species with different
detectability but similar observer exposure, environmental covariates that
distinguish habitat from accessibility, external effort proxies, and
independent structured survey data.

The safest wording is therefore not that the model removes bias. A better
statement is:

> The model attempts to separate shared observation effort from species-specific
> ecological structure, but the separation is only identifiable when the data
> contain enough variation to distinguish effort from habitat.

**Validation And Sensitivity Checks**

The practical safeguard is validation, not just architecture. Useful checks
include comparing predictions against structured survey data when available,
holding out spatial blocks that differ in observer accessibility, checking
whether correction removes known habitat associations, inspecting
species-specific residuals near high-effort habitats, and fitting sensitivity
models with weak, moderate, and strong effort correction.

The main point is:

> A neural IPPP is not especially compelling when it only replaces a classical
> smoother for one species. It becomes more interesting when used as part of a
> joint observation model that separates ecological intensity from observer
> effort by borrowing information across many species, locations, times, and
> observers.

## Heterogeneous graph neural networks for presence-only SDM

Reference: [Heterogeneous graph neural networks for species distribution modeling](https://arxiv.org/html/2503.11900v3).

This paper is a good example of a neural SDM idea that is meaningfully different
from a single-species neural IPPP. The model is not just:

\[
\lambda_j(s,z)=\exp(f_\theta(s,z))
\]

fit separately for each species. Instead, it represents the SDM problem as a
heterogeneous graph with two main node types:

- species nodes
- location nodes

Observed presence-only detections are represented as edges between species and
locations:

\[
(\text{species } j) \leftrightarrow (\text{location } s)
\]

The learning task is link prediction: given a species node and a location node,
predict whether a detection edge should exist between them.

This reframes SDM from a collection of independent single-species regressions
into a multi-species relational learning problem.

### Basic Graph Structure

The graph is bipartite in the simplest version:

\[
\mathcal{G}
=
(\mathcal{V}_{\text{species}},
\mathcal{V}_{\text{location}},
\mathcal{E}_{\text{detection}})
\]

Location nodes have environmental features such as extracted climate,
topographic, or other environmental covariates. Species nodes have species
features. In the paper's NCEAS benchmark experiments, species features are
mostly species identity, with group information available for some regions.

Each observed presence-only record creates a detection edge:

\[
e_{j,s}=1
\]

between species \(j\) and location \(s\). The model learns embeddings for
species and locations by message passing over this graph.

### Message Passing Intuition

The useful part is that species embeddings are informed by the environments
where they have been observed, and location embeddings can be informed by the
species observed there.

For example:

- a species node receives messages from all locations where that species was
  detected
- a location node can receive messages from all species detected at that
  location
- after message passing, the model scores possible species-location pairs

The predicted link probability can be thought of as:

\[
\Pr(e_{j,s}=1)
=
\sigma(h_j^\top h_s)
\]

where \(h_j\) is the learned species embedding, \(h_s\) is the learned location
embedding, and \(\sigma\) is the sigmoid function. The paper uses this kind of
dot-product decoder after graph message passing.

This gives the model a way to learn shared structure across species. If several
species occur in similar environments, their embeddings can become similar. If
two locations support similar species assemblages, their embeddings can also
become similar.

### Why This Is Different From A Single-Species IPPP/NIPPP

A single-species IPPP asks:

\[
\text{Given location features, how intense is species } j \text{ here?}
\]

The heterogeneous GNN asks:

\[
\text{Given the observed species-location graph, which species-location edges}
\]

\[
\text{are likely to be missing or plausible?}
\]

The neural advantage is therefore not just nonlinear function approximation. It
is relational inductive bias. The model can use co-occurrence structure,
species similarity, location similarity, and eventually other ecological
relationships if represented as graph edges.

This matters most when there are many species. Rare species may benefit from
information learned from more common species that occur in similar
environments or belong to related groups. A single-species model cannot borrow
that information unless the analyst manually builds a hierarchical model or
shared prior.

### Possible Extensions

The graph setup is flexible. In addition to species-location detection edges,
one could add:

- species-species edges for taxonomy, phylogeny, traits, competition,
  facilitation, or known co-occurrence
- location-location edges for spatial adjacency, distance, hydrologic
  connectivity, dispersal corridors, or environmental similarity
- observer-location or checklist-location edges for effort modeling
- dataset-source edges for GBIF institution, eBird protocol, or survey program
- temporal nodes or time-sliced location nodes for dynamic distributions

This is where the approach becomes more interesting than a plain NIPPP. It can
represent a richer ecological and observation graph instead of forcing
everything into one tabular covariate vector per location.

### Training With Presence-Only Data

The paper trains on presence-only detections, but it still needs negative
examples for the binary link-prediction loss. It constructs pseudo-negative
examples by assuming that unobserved species are absent at presence-only
locations and that all species are absent at sampled background locations.

So although the paper is framed as presence-only SDM, the training objective is
not a pure IPPP likelihood. It is a binary cross-entropy link-prediction
objective with sampled pseudo-negatives.

That distinction matters. The pseudo-negatives are not observed absences, and
their sampling strategy can affect the fitted model. The paper explicitly notes
that future work should explore alternative loss functions and weighting,
especially because optimizing over all edges equally can be affected by species
imbalance.

### Benchmark Setup

The paper evaluates on the NCEAS benchmark datasets, which provide
presence-only records for training and presence-absence records for testing.
The six regions are:

- Ontario, Canada
- South American countries
- Switzerland
- New South Wales
- Australian Wet Tropics
- New Zealand

Each region includes 11 to 13 pre-extracted environmental features, spatial
coordinates, and 10,000 background locations. Depending on the region, the task
includes roughly 20 to 54 species.

The paper reports that the GNN is better than prior benchmarked methods in
three regions, comparable in two, and worse in one. It is also compared against
a feed-forward multi-species MLP baseline. That comparison is important because
some gains may come from multi-species learning generally, not from the graph
architecture specifically.

### Why It Might Help In Practice

This approach is most compelling when:

- many species are modeled together
- some species are rare
- species have useful traits, taxonomy, or phylogenetic relationships
- locations have complex environmental or remote-sensing features
- species assemblage structure is informative
- observation data come from multiple sources with different biases
- the model needs to predict many species over many locations at once

For GBIF/eBird-style data, the graph could potentially encode not only species
and locations, but also observers, checklists, protocols, data sources, and
time. That would make the model a candidate for joint ecology/effort modeling,
not just species distribution prediction.

### Important Caveats

The paper is a proof of concept. It does not show that GNNs are universally
better than classical SDMs, nor that they solve the presence-only bias problem.

Several caveats matter:

- The benchmark species identities are obscured, so rich species traits are
  mostly unavailable.
- Environmental features are fixed, not time dynamic.
- Some detections may be old, with records older than 100 years and no recent
  records after 2006.
- The model still uses pseudo-negatives.
- The approach may overfit in regions with few unique test locations.
- Performance gains over prior methods partly reflect multi-species training,
  while many prior benchmark models were single-species models.

The clean interpretation is:

> Heterogeneous GNNs are promising for SDM when the problem is naturally
> relational: many species, many locations, shared environmental structure,
> species traits, co-occurrence, spatial connectivity, and observation-process
> metadata. They are less compelling if the task is only a single species with
> a small set of tabular covariates, where a classical IPPP, GAM, or spline
> model is already well matched to the problem.

For the blog post, this would fit as a stronger neural-network use case than a
single-species NIPPP. The model changes the structure of the learning problem
from smoothing one intensity surface to learning a species-location graph.
