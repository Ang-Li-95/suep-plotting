---
name: add-plots
description: >
  Add or modify histograms, selections, efficiency/ratio/profile plots, samples,
  or reweights in the suep-plotting framework. Use this skill whenever the user
  asks to plot a variable, study a distribution or a trigger/reconstruction
  efficiency, add a cut, change binning or labels, or add a sample. Everything
  is config-driven YAML (a variable catalog plus region trees), so consult this
  skill FIRST instead of reading the source code in src/. For running,
  submitting and plotting, use the run-suep skill.
---

# Adding histograms and plots in suep-plotting

A new plot needs **no Python**. What to fill is YAML:

- **variables**: `variables.yaml` at the repo root holds each variable once,
  with its expression, binning and axis title;
- **regions**: each set's `histograms.yaml` groups event-level cuts, then
  object-level cuts, then the variables to fill after both. A histogram is
  named `<region>/<cut>/<variable>`;
- **config sets**: `configs/<set>/` is one study, inheriting from the bases.

`custom/` builds the muon-system collections (LLPs, DBSCAN clusters). Read
`docs/configuration.md` for the full reference. Read `src/suep_plot/` only
when neither this skill nor the docs cover something.

## Look before you write

```bash
suep-config show  configs/configs_mds_signal --grep csc_cluster_size      # what the set fills, from which file
suep-config show  configs/configs_mds --section variables --grep llp      # the catalog as this set sees it
suep-config fields configs/configs_mds -s suep_mDark2_temp1 --grep cscCluster --ranges   # collections/fields after derive()
suep-config check configs/configs_mds                                     # after an edit: references, then every expression
```

- `show` and `check --static` read no input files.
- `fields` and `check` open one file per sample, so they need the grid proxy
  (see the run-suep skill).
- Checking a single set is fine on the login node. To check many sets, use
  `srun --partition=c --qos=rapid --time=00:30:00 --mem=8000 suep-config check configs/*/`.
- If `suep-config` is not on `PATH`, the installed entry points are older than
  it. Use `python -m suep_plot.cli config ...`. To install the new entry
  points, run `pip install -e .`, but ask the user first if the environment is
  shared.

## Config sets

Three sets are **bases, not studies**. They are marked `abstract: true` in
their `set.yaml`, and `suep-run` refuses them. A study extends one of them:

| base | holds |
|---|---|
| `common` | the reco-level cluster regions (`csc`, `dt`, cut `all`), their selections and derived plots; `rpc_mode: match`; the reco steps (`jerc, jet_id, clusters, cluster_isolation`) |
| `common_gen` | `common` plus everything that needs truth: the `llp` region, the matched/unmatched and isolation cuts, the efficiency chain, and the gen steps |
| `common_clustersrc` | `common` with every per-object cluster histogram cut to one named population (`<sys>_cluster_src`) |

| set | purpose |
|---|---|
| `configs_mds` | the MDS cluster study, **reco-only** fills; only `_extends: ../common` |
| `configs_mds_minpts50` | `configs_mds` with min cluster size 50: `set.yaml` + `columns.yaml` only |
| `configs_mds_signal` | the same plus **everything truth-dependent**, over the (mDark, T) grid |
| `configs_mds_gen` | `configs_mds_signal` plus the LLP shower-shape family; it **does** cluster |
| `configs_mds_data` | reco-only plus isolation cuts, on signal, DY and ZeroBias; no truth steps |
| `configs_mds_cosmics` | the cluster plots that need no prompt object; `steps: [clusters]` |
| `configs_mds_src_signal`, `_dy_jet`, `_dy_muon`, `_dy_iso`, `_zerobias`, `_cosmics` | **one study in six sets**: where a cluster comes from. Each set fills one population; they are overlaid as six samples |
| `configs_mds_objects` | prompt objects (post-JERC PUPPI MET, HT, jets, leptons) and cluster multiplicities; its own histograms and selections (only `columns.yaml` extends `common`) |
| `configs_mds_sigonly`, `configs_mds_trigger`, `configs_g4compare` | the two signal points only; HLT/L1 trigger efficiencies; pre/post Geant4 fix (its inputs are gone from EOS) |
| `configs_mds_rpcmerge`, `_rpcmerge_data` | the signal / data studies with `rpc_mode: merge` |
| `configs_smoke` | not a study: one file, two histograms, run by `scripts/smoke_test.sh` |

Each set can describe itself in one line, the `description:` in its
`set.yaml`. A wrong `-c` lists these descriptions.

### Files of a set

The files are `set.yaml`, `samples.yaml`, `histograms.yaml`, `selections.yaml`,
`corrections.yaml`, `reweights.yaml`, `derived_plots.yaml`, `columns.yaml` and
`variables.yaml`.

- A directory is a set if it has a `histograms.yaml` or a `set.yaml`.
- A missing file is empty, or, when `set.yaml` has `extends: <parent>`, it is
  the parent's file.

**Base a new study on an existing one with `extends`; do not copy it.** A
copied set and its original drift apart the next time someone adds a plot to
one of them.

- Write a `set.yaml` with `extends: ../<parent>` and a `description:`.
- Add only the files that differ. Start each one with `_extends: ../<parent>`
  so it changes part of the parent's file instead of replacing it.
- Mappings merge deeply, lists are replaced, and `name: null` deletes an
  inherited definition.

`configs/configs_mds_minpts50/` is the model:

```yaml
# set.yaml
extends: ../configs_mds
description: "configs_mds with the standard MDS clustering: min cluster size 50"
# columns.yaml
_extends: ../configs_mds
parameters: {cluster_min_samples: 50}
```

Datasets are defined once, in the top-level `datasets.yaml`. A `samples.yaml`
is `_registry: ../../datasets.yaml` plus the names to use, optionally with
per-set overrides (`name: {color: ...}`), or with `_from:` to take an entry
under another name. Add a new dataset to `datasets.yaml`, not to a config. A
`files:` entry may be a directory, local or xrootd, which is walked for
`*.root`. `files_per_job:` there sets that dataset's Slurm shard size.

## Fill time or plot time: decide before running anything

- **Fill time** needs `suep-run` again. This covers:
  - a variable's `expression` and binning;
  - adding a variable to a region;
  - `evt_sel`, `cut`, `selections` and `weight`;
  - anything in `custom/` or in `columns.yaml`.
- **Plot time** needs only `suep-plot` again (seconds, no xrootd). This covers:
  - every `label` (the variable's, the region's, the cut's), `annotation`,
    `rebin`, `flow`, `blind` and `log_*`;
  - sample colours and labels;
  - **everything in `derived_plots.yaml`** (efficiencies, ratios, profiles,
    projections, `spans`).

`suep-plot` re-reads this styling from the config set each time.

## Recipe: plot a variable

**1. The variable.** Look it up (`suep-config show <set> --section variables
--grep ...`). If it does not exist, add one line to `variables.yaml`:

```yaml
csc_cluster_size_over_r: {expression: "events.cscCluster.size / events.cscCluster.r", bins: [50, 0, 2], label: 'CSC cluster $N_{\mathrm{hits}}/r$ [1/cm]'}
```

- Allowed keys:
  - `expression`;
  - `bins: [n, lo, hi]` or `edges: [...]`;
  - `label`;
  - the optional 1D display defaults `log_x`, `log_y`, `flow`, `rebin` and
    `blind`.
- Any other key is an error.
- Names follow the catalog's pattern (`csc_cluster_*`, `dt_cluster_*`,
  `llp_*`, ...), and CSC and DT each get their own line.
- A set can retune a variable for all of its histograms in
  `configs/<set>/variables.yaml`, for example `llp_decay_r: {bins: [90, 0, 900]}`.
  A variable that only one set uses can live there too.

**2. Where to fill it.** List it under a region and a cut in `histograms.yaml`.
Put it in a base to get it in every study built on that base:

```yaml
regions:
  csc:
    objects:
      all:      {vars: [..., csc_cluster_size_over_r]}
      iso_muon: {cut: csc_cluster_iso_muon, label: muon-isolated, vars: [csc_cluster_size_over_r]}
```

This fills `csc/all/csc_cluster_size_over_r` and
`csc/iso_muon/csc_cluster_size_over_r`. The histogram name is also the figure's
folder: `plots/csc/iso_muon/csc_cluster_size_over_r.png`.

**`vars` lists are replaced, not appended**, when a set extends another. To add
to an inherited cut, edit the list where it is defined. The bases use YAML
anchors (`vars: &csc_all`), so one edit there reaches every cut that reuses the
list. Alternatively, restate the whole list in the child.

**3. Check, then run:** `suep-config check <set>`, then a one-file `suep-run`
(see the run-suep skill).

### Entries in a `vars` list

| Entry | Histogram |
|---|---|
| `csc_cluster_size` | the catalog variable, 1D |
| `[llp_decay_absz, llp_decay_r]` | 2D; x and y from the catalog; leaf `llp_decay_absz_vs_llp_decay_r` |
| `{var: llp_pt, bins: [50, 0, 50], log_y: true}` | 1D with overrides (any histogram key) |
| `{name: llp_decay_rz, x: llp_decay_absz, y: {var: llp_decay_r, bins: [60, 0, 900]}}` | 2D with its own leaf name and a per-axis override |
| `{var: csc_cluster_rpc_bx, selections: [csc_cluster_has_rpc]}` | an extra cut, or a `weight:`, for this histogram only |

If the same variable appears twice under one cut, one of the two entries needs
a `name:`. `log_z: true` on a 2D entry gives a log colour scale.

### What a region means

```yaml
regions:
  csc:                        # region
    label: CSC clusters       # on the figures: "CSC clusters · muon-isolated"
    evt_sel: [has_csc_cluster] # event-level cuts, AND-ed (optional)
    evt_vars: [csc_cluster_n]   # filled once per event -> csc/csc_cluster_n
    objects:
      all: {vars: [...]}      # no cut; its label is never drawn
      iso_muon:
        cut: csc_cluster_iso_muon   # one selection or a list, AND-ed
        label: muon-isolated
        vars: [...]
```

- A histogram's selections are the region's `evt_sel`, then the cut's `cut`,
  then the entry's own `selections`.
- **Per object or per event is read off the expression.**
  - A jagged variable (`events.cscCluster.size`) fills once per object that
    passes.
  - A per-event variable takes event-level selections only; an object-level
    one there is an error. For "has a passing object", define an event-level
    selection: `has_x: "ak.any(events.<object_sel>, axis=1)"`.
  - Never write `per_object`.
- `<cut>: null` in a child deletes an inherited cut, and `evt_vars: []`
  clears an inherited list.
- A per-system family of regions can be written once with a `_repeat` inside
  `regions:`, using the file's `_axes`. `<SYS>` / `<SYSU>` are substituted in
  every string. The sigil is `<...>` because the labels are full of LaTeX
  braces.

## Recipe: a new cut

A selection is one line in `selections.yaml`. Event or object level is read
off the mask:

```yaml
csc_cluster_big: "events.cscCluster.size > 100"          # one entry per cluster -> object level
has_two_csc_clusters: "ak.num(events.cscCluster) >= 2"     # one entry per event   -> event level
```

An object-level cut becomes a new entry under `objects:`. An event-level one
becomes a region's `evt_sel:`, often in a new region:

```yaml
regions:
  csc:
    objects:
      big: {cut: csc_cluster_big, label: '$N_{\mathrm{hits}} > 100$', vars: [csc_cluster_eta, csc_cluster_phi]}
  csc_2cl:
    label: '$\geq 2$ CSC clusters'
    evt_sel: [has_two_csc_clusters]
    evt_vars: [csc_cluster_n]
    objects:
      all: {vars: [csc_cluster_size]}
```

- A cut must be over the same collection as the variables under it. A CSC cut
  on a DT variable stops the run on the first chunk, naming the histogram and
  sample. The startup check evaluates each expression on its own, so it does
  not catch this case.
- A selection needs the mapping form only for a `label` or a `collection`:
  `{expression: ..., collection: Muon, label: ...}`. A `level:` key is
  accepted but must agree with the mask.
- Every selection is also a row in `cutflow.txt`/`.csv`. The counts are
  independent, not sequential.

## Recipe: efficiency, ratio, profile (`derived_plots.yaml`)

These are computed at plot time from filled histograms. The region-aware form
writes one entry for a list of variables:

```yaml
matchfrac_csc:
  type: efficiency                 # Clopper-Pearson intervals, one curve per sample
  region: csc
  numerator: matched               # cuts of that region
  denominator: all
  label_y: fraction of CSC clusters LLP-matched
  vars: [csc_cluster_size, csc_cluster_eta]
  spans:                           # optional shaded x ranges
    - {lo: 582, hi: 623, label: 'ME1/1'}
```

This expands to `csc/matchfrac_csc/csc_cluster_size`, which divides
`csc/matched/csc_cluster_size` by `csc/all/csc_cluster_size`.

- Both cuts must fill the variables you list, with identical binning.
- `vars` names leaves, so a 2D source is `x_vs_y` or its `name:`. It gives a
  2D efficiency map; add `label_z`.

| `type` | reads | draws |
|---|---|---|
| `efficiency`, `ratio` | `numerator`, `denominator` | Clopper-Pearson efficiency / plain ratio |
| `profile_x`, `profile_y`, `projection_x`, `projection_y` | `cut:` (region-aware) or `source:` (a 2D histogram) | mean of y per x bin, or a projection |
| `sig_vs_bkg` | `signal`, `background` | signal samples of one histogram against background samples of another, unit-normalized |

- For single-source types, leaving out `cut:` means the region's per-event
  histogram.
- The explicit form names whole histograms, for example
  `numerator: hlt_csccluster_loose/evt_max_csc_cluster_size`.
- A source that is missing from the pickles is warned about and skipped.

**Never pass `--lumi` to efficiency plots**, or to any MDS set. `xs: 1.0` there
is a placeholder, and lumi scaling distorts the binomial intervals.

## Expression language

Names in scope:

- `events` (NanoEvents; also `ev`);
- `ak`, `np`;
- the builtins `abs`, `len`, `min`, `max`, and no others.

NanoAOD naming: `HLT_X` becomes `events.HLT.X`, and `PFMET_pt` becomes
`events.PFMET.pt`.

Idioms:

- `ak.num(coll)`, `ak.sum(x, axis=1)`;
- `ak.firsts(x)`, the None-safe leading object;
- `coll.nearest(other).delta_r(coll)`.

Missing values (`None`, NaN) are dropped while filling, with the weights kept
aligned.

**Every selection is also a column with its own name**, so an expression can
use the mask instead of restating the cut:

| want | expression |
|---|---|
| how many objects pass | `ak.sum(events.electron_sel, axis=1)` |
| does any object pass | `ak.any(events.electron_sel, axis=1)` |
| the selected objects | `events.Jet.pt[events.jet_sel]` |
| an event-level mask | `events.high_met` (already flat) |

- The masks are attached after `custom/`, so masks over derived collections
  (`events.cscCluster.matched`) work.
- A selection whose name is already a field of `events` is skipped, with a
  message.

**Lambdas and comprehensions work.** The namespace is passed to `eval` as
globals, so nested scopes see `ak` and `np`; see
`histograms._compile_expr` if `NameError: name 'ak' is not defined` ever
returns. This lets you name an intermediate result instead of writing it
twice. For example, the maximum |Δφ| over all cluster pairs:

```yaml
expression: >-
  (lambda p: ak.max(abs((p.a - p.b + np.pi) % (2 * np.pi) - np.pi), axis=1))(
  ak.combinations(events.cscCluster.phi, 2, axis=1, fields=["a", "b"]))
```

`ak.combinations` makes "fewer than 2 objects" come out as `None`, which is
dropped, instead of a spurious 0.

## Errors are fatal, and that is the point

- Config mistakes stop the run before any file is opened, and the message
  names the file, often with a "did you mean". Examples:
  - an unknown variable, selection, key or `_` directive;
  - invalid YAML;
  - a bad `columns.yaml`.
- Every expression is evaluated on the first events of every sample's first
  file before processing. A failure, there or later on some chunk, stops the
  run and names the histogram or selection, the sample and the expression.
- A histogram that cannot fill on one of the set's samples (a branch that
  sample lacks, or a collection whose `columns.yaml` step is off) belongs in a
  set without that sample, or the step must be enabled. **Don't work around
  it.**

## Removed forms (an error since 2026-11-10)

- Histograms outside `regions:` and the long form (`expression`, `bins`,
  `lo`, `hi` written on a histogram). Every histogram is a region entry that
  names catalog variables; a new expression goes into `variables.yaml`.
- An inline axis in an entry (`x: {expression: ...}`).
- `variants:` and `_variant_sets`: write an object cut in the region instead.
- An explicit `per_object`.

The shipped configs were converted to regions on 2026-11-09.
`docs/histogram-renames.tsv` maps every old histogram name to its
`region/cut/variable` name, per set. Pickles filled before then still carry
the old names.

## Derived collections (`custom/`, on MDSNano samples)

- `events.llp` has one entry per gen LLP (SUEPGenPart pdgId 999999) with the
  fields `pt eta phi mass energy decayR decayZ Llab betagamma ctau inCSC inDT
  inRPC nHitsCSC nHitsDT nHitsRPC nHitsTotal recoCSC recoDT recoRPC reco gidx
  lidx`. It exists only in sets with the gen steps (`common_gen` and below).
- `events.cscCluster` and `events.dtCluster` are DBSCAN clusters of rechits,
  with the fields:
  - `size eta phi x y z r etaSpread phiSpread rSpread zSpread nStation
    stationSpan avgStation maxStationFrac`, the `nLayer...` / `nZLayer...`
    families, `nMatchedHits matchedLLPIdx matched hasTruth time` (time: CSC
    only in practice);
  - the RPC content `nRPCHits`, `rpcBx`, `rpcBxSpread`, ...;
  - `drMuon` / `drJet`: the dR from the cluster centroid to the closest
    *selected* Muon / Jet. The value is 999 (`NO_OBJECT_DR`) when the event
    has none, so isolation is a plain `events.cscCluster.drMuon >= 0.4`.
- **There is normally no `events.rpcCluster`.** `rpc_mode: match` (the
  default in `common`) attaches the RPC rechits to the CSC and DT clusters.
  `rpc_mode: separate` creates a third collection, and `merge` clusters the
  RPC rechits together with the CSC / DT ones.
- **"Selected" for the cluster dR is an ordinary selection.** It is defined in
  the set's `selections.yaml` under the fixed names `muon_sel_foriso` and
  `jet_sel_foriso` (`custom/helpers.py:ISO_SELECTIONS`), each with a
  `collection:`. A set that enables `cluster_isolation` without them fails at
  startup. Each one's multiplicity is also attached, as
  `events.n_muon_sel_foriso` and `events.n_jet_sel_foriso`.
- **Jets:**
  - 2024 NanoAOD has no `Jet_jetId`, so the `jet_id` step recomputes it into
    `events.Jet.tightId` and `.tightLepVetoId`. Use these columns; there is no
    `jet_id()` function.
  - The `jerc` step applies JEC to `events.Jet` in place, with JER smearing on
    MC only, and rebuilds the Type-1 `PuppiMET`. `events.Jet.pt` and
    `events.PuppiMET.pt` are therefore already corrected.
- **Fill-time knobs** live in the set's `columns.yaml`:
  - `parameters:` takes `cluster_eps`, `cluster_min_samples`,
    `rpc_min_samples`, `rpc_mode`, `match_min_hits`, `oot_time_cut`,
    `pair_max_hits`, `dr_quantiles`, `llpidx_convention`, `jerc`, `jerc_era`,
    `jerc_algo` and `jerc_data_tag`;
  - `steps:` takes `jerc, jet_id, clusters, cluster_isolation, llp, llp_hits,
    llp_reco, llp_shape`;
  - `custom/params.py:PARAM_SPEC` is the source of truth;
  - there are no `MDS_*` environment variables.
- **Truth:**
  - On samples without truth branches (central background MDSNano),
    `matched` and `hasTruth` are always False.
    `matched | ~events.<c>.hasTruth` therefore selects signal-like clusters
    uniformly.
  - Gen-matched CSC rechits per event: `ak.sum(events.cscRechits.llpIdx >= 0,
    axis=1)`. Never sum `events.llp.nHitsCSC`, which double counts.

## Branches (2024 signal MDSNano, verified)

- Triggers:
  - `HLT_CscCluster_{Loose,Medium,Tight}`;
  - `HLT_DoubleCscCluster{75,100}`;
  - `HLT_PFMETNoMu1{2,3,4}0_PFMHTNoMu1{2,3,4}0_IDTight` (+ `_FilterHF`).
- Objects: `PFMET_*`, `PuppiMET_*`, `Jet`, `Muon`, `Electron`, `SUEPGenPart`.
- Muon-system hits: `cscRechits` (`llpIdx` points into SUEPGenPart),
  `dtRecHits`, `rpcRecHits`.
- HLT clusters: `cscMDSHLTCluster`, `dtMDSHLTCluster`.

Central MDSNano v2 collapses `cscRechits_Z` to one value per chamber; the
private signal keeps 6 planes per chamber. `suep-config fields` shows what a
given sample really has.

## Deep dives (read only when needed)

- `docs/configuration.md`: every file and directive (`_extends`, `_repeat`,
  `_include`, `_registry`, `set.yaml`), the catalog, regions, corrections,
  reweights and derived plots.
- `docs/derived-columns.md`: `custom/`, with the collections field by field,
  the parameters, `rpc_mode`, and what each step costs.
- `docs/mds-study.md`: the physics of the MDS cluster study, the six cluster
  populations, and the event display.
- `docs/known-issues.md`: clustering memory grows as N² per event.
- `docs/histogram-renames.tsv`: old to new histogram names.
