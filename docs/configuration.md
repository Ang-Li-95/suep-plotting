# Configuration reference

A **config set** is a directory of YAML files describing one study. `suep-run`
and `suep-plot` take it with `-c`:

```
configs/configs_mds_signal/
├── samples.yaml         # which datasets to run, cross sections, styling
├── histograms.yaml      # histogram definitions (NanoEvents expressions)
├── selections.yaml      # named event-/object-level cuts
├── corrections.yaml     # correctionlib scale-factor definitions
├── reweights.yaml       # event-/object-level reweighting
├── derived_plots.yaml   # profiles/projections/efficiency/ratio at plot time
└── columns.yaml         # parameters + steps of derive() -- see derived-columns.md
```

Every one of them is read by `src/suep_plot/config.py`, so the directives in the
next section mean the same thing in all of them.

---

## Directives

These four keys are understood in **every** config file. They are directives,
not definitions: like any `_`-prefixed key they never become a histogram, a
selection or a sample.

### The two bases

`configs/common/` and `configs/common_gen/` are config *sets* that no run uses
directly. `common` holds the reco-level cluster histograms and selections every
study fills; `common_gen` `_extends` it and adds the gen-level ones — the LLP
collection, the truth splits, the efficiency chain and its derived plots.

```
configs/common/            reco histograms + selections + derived plots,
  │                        rpc_mode: match, the four reco steps
  ├── configs_mds/                  _extends ../common      (and nothing else)
  ├── configs_mds_data/             _extends ../common + the isolation splits
  ├── configs/common_clustersrc/    + every cluster histogram cut to one named
  │     └── configs_mds_src_*/        population; the five sets define what it is
  └── configs/common_gen/  + LLP collection, truth splits, efficiency chain,
        │                    and the four gen steps
        ├── configs_mds_signal/     _extends ../common_gen  (and nothing else)
        └── configs_mds_rpcmerge/   _extends ../configs_mds_signal, rpc_mode: merge
```

`common` carries only the **reco** steps (`jerc, jet_id, clusters,
cluster_isolation`); `common_gen` adds the four gen ones. A study with no
gen-level fills should not pay for them, and `events.llp` being absent is what
makes a stray gen expression fail at validation instead of filling empty.

`common` also fixes **`rpc_mode: match`** as the default, so there is no
standalone `events.rpcCluster` and no `rpc_cluster_*` histogram anywhere: the
RPC rechits are attached to the finished CSC and DT clusters, which carry their
RPC content and timing. Everything per-system is therefore a CSC/DT pair — the
`systems` axis in `common/selections.yaml` has two entries, and `common_gen`
inherits it.

A study attaches its own splits by overriding **one key** of an inherited
histogram — `csc_cluster_size: {variants: csc_variants}` — because mappings
merge deeply. That is why `common` deliberately defines no `_variant_sets` and
puts no `variants:` on anything: what a cluster is split by is the study's
business, and `configs_mds` wants the inclusive fills on their own.

### `_extends` — inherit another config set

```yaml
_extends: ../configs_mds_signal

cluster_size_rpc: null            # drop an inherited definition
cluster_size_csc:                 # override part of one (mappings merge deeply)
  hi: 800
```

The same-named file from the other set is loaded first and this file's keys are
merged over it. `null` deletes. Chains are followed, and a cycle is an error.

This is how the RPC-mode studies are defined: `configs/configs_mds_rpcmerge/`
is `configs_mds_signal` with `rpc_mode: merge`, a handful of deletions and the
RPC timing histograms added — so the two stay comparable plot for plot by
construction, instead of being kept in step by hand.

### `_repeat` — write a per-system family once

Most definitions in this analysis exist once per muon subdetector. `_repeat`
expands one template over a list of substitutions:

```yaml
_axes:
  systems:
    - {SYS: csc, SYSU: CSC}
    - {SYS: dt,  SYSU: DT}
    - {SYS: rpc, SYSU: RPC}

_repeat:
  - over: systems                 # or an inline list of mappings
    defs:
      llp_in_<SYS>:
        level: object
        expression: "events.llp.in<SYSU>"
```

- The sigil is `<NAME>`, **not** `{NAME}` — the labels are full of LaTeX braces.
- Substitution reaches keys and every string leaf, including inside lists.
- A value that is *exactly* one placeholder keeps the substituted **type**, so
  `hi: <HI>` with `HI: 500` stays the integer 500 and `edges: <EDGES>` can carry
  a whole list. Embedded in text (`label: "up to <HI> GeV"`) it interpolates as
  a string.
- There are no conditionals: a family that covers only CSC and DT lists only
  those two.
- A generated name colliding with an explicit definition is an error.

Per-system definitions that genuinely differ — different expressions, different
binning per system — are better written out; a `_repeat` whose axis has to carry
a dozen hoisted values is harder to read than the thing it replaced.

### `_include` — merge in a shared fragment

```yaml
_include: [../_common/shared_cuts.yaml]
```

Definitions from the named files are merged in, then this file's own keys are
merged over them. Unlike `_extends`, which follows the config *set*, an include
names one file — reach for it when a handful of definitions are shared by sets
that are siblings rather than variants of each other.

No config set uses it today: the prompt-object cuts that were its one candidate
are written out in each `selections.yaml` instead, so a cut can be retuned in
one study without silently moving the others (see
[derived-columns.md](derived-columns.md#prompt-objects-live-in-selectionsyaml)).

> The up-to-date check follows `_include`, `_extends` and `_registry` (see
> `config.config_sources`), so editing an included fragment does reprocess every
> set that reads it — no `--force` needed.

### `_registry` — select from a pool

What `samples.yaml` does with the central dataset registry:

```yaml
_registry: ../../datasets.yaml

suep_mDark2_temp1:                     # take the registry entry verbatim
suep_mDark2_temp2: {color: "tab:red"}  # ... with these keys overridden
signal_alias:
  _from: suep_mDark2_temp1             # base it on a differently-named entry
  label: "the same data, another name"
```

`_registry` and `_include` deliberately mean opposite things for an entry with
no value: an include *merges*, so `name:` deletes; a registry is a pool you
*select* from, so `name:` takes. A file with no `_registry` needs every
definition spelled out.

---

All behaviour is controlled by the YAML files of a config set under `configs/`.

### samples.yaml

```yaml
suep_mMed125_mDark2:
  files:                                    # paths, globs, directories, or xrootd URLs
    - "/path/to/NanoAOD/*.root"             # a directory is walked recursively for *.root
  tree: Events                              # TTree name (default: Events)
  is_data: false                            # true for collision data
  xs: 1.0                                   # cross section in pb (MC; used by --lumi scaling)
  label: "SUEP $m_{Med}$=125"               # legend label (LaTeX ok)
  color: "tab:blue"                         # matplotlib color
  linestyle: "--"                           # optional: dash pattern for step curves
  group: signal                             # signal / background (data via is_data)
  scale: 100                                # optional: draw signal x100 (legend shows "×100")
```

A `files:` entry may be a file, a glob (`*`/`?`), or a **directory** — local or
xrootd — in which case it is walked recursively for `*.root`. `group` drives plot
styling: `background` → stacked fill, `signal` → step overlay (or fill if alone),
`is_data: true` → error bars.

#### Central dataset registry

Datasets are defined once in the top-level [`datasets.yaml`](../datasets.yaml); each
config picks the combination it needs instead of repeating the definitions:

```yaml
_include: ../../datasets.yaml    # one path or a list; relative to this samples.yaml

suep_mDark2_temp1:            # registry entry, verbatim
suep_mDark2_temp2:            # registry entry with per-study overrides
  color: "black"
  scale: 100
dy:                           # same dataset under a different config name
  _from: dy_central_50to120
qcd_private:                  # full standalone definition — registry not required
  files: ["/my/private/dir"]
  xs: 1.0
```

Per-sample keys are merged over the registry entry (config wins). A name that is
neither in the registry nor carries its own `files:` is an error listing the
available datasets, so typos fail at startup rather than mid-run. Configs that
define everything inline keep working unchanged — `_include` is optional.
`sumw` (Σ genWeight) and `nevents` are recorded per sample in the pickle; when
`--lumi` is passed to the plotter, each MC sample is normalized by
`xs × lumi × 1000 / sumw` (data is never scaled; MC without `xs`/`sumw` is left
raw with a warning).

### histograms.yaml

**The main file you edit to add plots.** Expressions use NanoEvents object syntax.

```yaml
jet_pt:
  expression: "events.Jet.pt"               # per-jet pt (jagged)
  bins: 60
  lo: 0
  hi: 600
  label: "$p_{T}^{jet}$ [GeV]"
  per_object: true                          # flatten jagged -> one entry per jet
  selections: [baseline, high_ht]           # AND of these cuts (optional)
  weight: "events.someScalarBranch"         # extra per-event weight (optional)
```

Fill-time fields (changing them requires re-running `suep-run`):

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `expression` | yes | — | NanoEvents expression (see [Expression language](#expression-language)). |
| `bins` / `lo` / `hi` | yes* | — | Uniform axis binning. |
| `edges` | yes* | — | Explicit bin edges for variable binning, e.g. `[0, 20, 40, 80, 160]` (alternative to `bins/lo/hi`). |
| `per_object` | no | `false` | If `true`, expression returns a jagged array (e.g. one value per jet); it is flattened, with per-event weights repeated per object. Otherwise it must return one value per event. |
| `selections` | no | `[]` | Selection names from `selections.yaml`, AND-ed. |
| `weight` | no | — | Extra per-event weight (× genWeight × corrections). |
| `variants` | no | — | Repeat this histogram under extra selections — see [`variants`](#the-same-plot-under-several-selections-variants). |

Plot-time fields (take effect on the next `suep-plot`, **no reprocessing**):

| Field | Default | Description |
|-------|---------|-------------|
| `label` | name | X-axis label (LaTeX between `$…$`). |
| `blind` | `false` | Don't draw data for this histogram (signal regions). |
| `rebin` | — | Merge N adjacent bins at plot time (1D only). |
| `flow` | `sum` | Under/overflow is folded into the first/last bin, so nothing is lost off the axis; `none` drops it instead. |
| `log_x` / `log_y` | `false` | Logarithmic axes for this histogram. |
| `log_z` | `false` | Log color scale (2D histograms). |

**2D histograms** use `expression_x`/`expression_y` with
`bins_x/lo_x/hi_x/bins_y/lo_y/hi_y` (or `edges_x`/`edges_y`) and
`label_x/label_y`. They render as colz and can feed the derived
profile/projection plots.

#### The same plot under several selections (`variants`)

Cluster studies usually want every variable under the same handful of cuts
(matched / unmatched, in-time / out-of-time, per station …). Write the variable
once and list the cuts — the loader expands the (variable × selection) matrix:

```yaml
_variant_sets:                  # reusable; top-level '_' keys are not histograms
  csc_match:
    matched:   {selections: [csc_cluster_matched],   label_prefix: "matched "}
    unmatched: {selections: [csc_cluster_unmatched], label_prefix: "unmatched "}

csc_cluster_size:
  expression: "events.cscCluster.size"
  per_object: true
  selections: [has_csc_cluster]     # base cuts: kept by every variant
  bins: 50
  lo: 0
  hi: 500
  label: 'CSC cluster $N_{\mathrm{hits}}$'
  variants: csc_match               # or an inline mapping of the same shape
```

That yields three histograms — `csc_cluster_size` (base cuts only),
`csc_cluster_size_matched` and `csc_cluster_size_unmatched` — and adding a third
cut to `csc_match` adds one plot to *every* variable that references it.

| Variant key | Effect |
|---|---|
| `selections` | **Appended** to the base `selections`, so base cuts keep applying. |
| `label_prefix` / `label_suffix` | Decorate the base `label` (legends and axis titles stay distinguishable). |
| any other key | Overrides the base outright — e.g. `log_y: false`, a different `hi`. |
| `keep_base: false` | *On the histogram, not the variant:* drop the unselected version and keep only the variants. |

Expansion happens where the YAML is read, so variants behave exactly like
hand-written entries: they appear in `cutflow.txt`, the `index.html` gallery and
`derived_plots.yaml` under their expanded names (`<name>_<variant>`), and their
selection names are validated the same way. A variant that would overwrite an
existing histogram, or an unknown variant-set name, is a fatal config error.

### selections.yaml

Named boolean cuts referenced by histograms.

```yaml
baseline:                                    # event-level
  label: "Baseline"
  expression: "ak.num(events.Jet) >= 1"

good_jets:                                   # object-level
  label: "$p_T^{jet} > 30$, $|\\eta| < 2.4$"
  expression: "(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4)"
  level: object
```

| Field | Required | Description |
|-------|----------|-------------|
| `expression` | yes | Boolean array. Event-level → one per event; object-level → jagged, matching a collection. |
| `level` | no | `event` (default) or `object`. |
| `label` | no | Human-readable label. |

**Object-level** selections filter which objects enter a `per_object` histogram;
applied to an event-level histogram they mean "require ≥1 passing object"
(auto `ak.any(…, axis=1)`). Multiple selections on one histogram are AND-ed.

Pre-defined: `baseline`, `high_ht`, `has_muon`, `has_csc_cluster`,
`has_dt_cluster`, `met_gt50` (event-level) and `good_jets` (object-level).

### corrections.yaml

Scale factors, applied via coffea's tooling. All entries optional; an empty file
means no corrections. Data samples are always skipped.

```yaml
pileup:
  file: "auto:LUM/2024_Summer24/puWeights.json.gz"
  name: "Collisions2024_goldenJSON"
  inputs: ["events.Pileup.nTrueInt", "'nominal'"]
  kind: event_weight
  apply_to: all
```

| Field | Required | Description |
|-------|----------|-------------|
| `file` | yes | correctionlib `.json(.gz)`, or `auto:POG/year/file` (resolved from `$CORRECTIONLIB_DATA`, then cvmfs jsonpog-integration). |
| `name` | yes | Correction name inside the JSON. |
| `inputs` | yes | Expressions passed to the evaluator **in the order the correction declares its inputs**. Bare string literals like `"'nominal'"` select a systematic/category axis. |
| `kind` | yes | `event_weight` (one weight/event) or `object_sf` (one SF/object; per-event weight = product over objects). |
| `apply_to` | no | `all` (default) or a list of sample groups. |

For `object_sf`, jagged (per-object) inputs are flattened automatically and scalar
inputs (the systematic string) are passed through, so the correctionlib string
axis works correctly; the per-event weight is
`ak.prod(ak.unflatten(sf, counts), axis=1)`. Each correction is registered into a
`coffea.analysis_tools.Weights` object, so up/down systematics are a natural
extension (`weights.add(name, nominal, up, down)`).

### reweights.yaml

Analysis-level **reweighting** for studies — flatten a spectrum, match MC
kinematics to data, emulate a trigger turn-on. Weights multiply into the event
weight alongside genWeight and `corrections.yaml` scale factors (all via
coffea `Weights`), so they also show up in cutflows and `--lumi` scaling.
Config errors here are fatal (they change physics results).

Two kinds, auto-detected:

```yaml
soft_met_weight:                       # expression -> weight per event
  expression: "1.0 / (1.0 + events.PuppiMET.pt / 200.0)"
  apply_to: [background]

jet_eta_flat:                          # object-level: product over objects
  expression: "1.0 + 0.05 * abs(events.Jet.eta)"
  level: object

ht_shape:                              # binned map, looked up in `variable`
  variable: "ak.sum(events.Jet.pt, axis=1)"
  edges: [0, 100, 200, 400, 800, 2000]
  weights: [1.25, 1.10, 1.00, 0.90, 0.75]
  clamp: true
  apply_to: [background]

ht_dataMC:                             # map file generated by suep-reweight
  file: ht_map.yaml
  apply_to: [background]
```

| Field | Default | Description |
|-------|---------|-------------|
| `expression` | — | NanoEvents expression → weight (event-level scalar, or jagged with `level: object`). |
| `variable` (+ `edges`, `weights`) | — | 1D binned lookup. 2D: `variable_x/variable_y`, `edges_x/edges_y`, nested `weights`. |
| `file` | — | Load map fields from another YAML (relative to the config dir); entry keys override. |
| `level` | `event` | `object`: one weight per object, event weight = product over objects (empty events → 1). |
| `apply_to` | `all` | Sample groups to reweight (MC only). |
| `samples` | — | Explicit sample list (overrides `apply_to`; may include data). |
| `clamp` | `false` | Binned maps: out-of-range values use the nearest bin (default: weight 1 outside). |
| `fill_only` | `false` | Don't apply globally; only used where a histogram sets `weight: "@<name>"`. |

**Per-object fill weighting:** a histogram's `weight` may be a jagged
expression or a `"@<map name>"` reference. For `per_object` histograms each
object then carries its own weight (e.g. reweight the jet-pT spectrum and look
at other jet variables); for event-level histograms the per-object weights are
multiplied into one weight per event. Mark such maps `fill_only: true` or the
weight is applied twice (the processor warns).

**Deriving a map from processed samples** — the classic "reweight MC to match
data in X" study:

```bash
suep-run -c configs/configs_mds -o output                 # 1. fill histograms as usual
suep-reweight output/ --hist ht \
    --num data_2024 --den qcd \
    -o configs/configs_mds/ht_map.yaml        # 2. map = shape ratio data/MC
# 3. reference it in configs/configs_mds/reweights.yaml (ht_dataMC above), then
suep-run -c configs/configs_mds -o output                 # 4. only affected samples re-run
```

The generated map stores the histogram's own fill expression, bin edges and
the (by default normalized, shape-only) ratio; `--no-normalize` keeps the
absolute ratio, `--no-clamp` gives weight 1 outside the map range. 2D
histograms produce 2D maps. Bins with an empty denominator get weight 1.

### derived_plots.yaml

Computed at **plot time** from saved histograms — no reprocessing. Types:
`profile_x`/`profile_y` (from a 2D hist), `projection_x`/`projection_y`,
`efficiency` (Clopper–Pearson errors), `ratio`. See the file header for fields.

---

## Expression language

Expressions in `histograms.yaml`, `selections.yaml`, `corrections.yaml`,
`reweights.yaml`, and any `weight` are `eval`'d with these names in scope
(histogram `weight` fields may instead reference a reweight map as
`"@<name>"`):

- `events` — the coffea NanoEvents array (also aliased `ev`)
- `ak` — awkward, `np` — numpy
- builtins `abs`, `len`, `min`, `max`

Because objects carry NanoAOD behaviours you get vectors and helpers for free:

```python
events.Jet.pt                                  # jagged per-jet pt
ak.num(events.Muon)                            # muons per event
ak.sum(events.Jet.pt, axis=1)                  # scalar HT
abs(events.Jet.eta) < 2.4                      # object-level mask
ak.firsts(events.Jet.pt)                       # None-safe leading jet
(events.Muon[:, 0] + events.Muon[:, 1]).mass   # dimuon mass (needs ≥2 muons)
events.Jet.nearest(events.Muon).delta_r(events.Jet)   # ΔR to nearest muon
```

Collections in the MDSNano file include `Jet`, `Muon`, `Electron`, `PuppiMET`,
`GenMET`, `Pileup`, `PFCand`, `SUEPGenPart`, `cscMDSHLTCluster`, `dtMDSHLTCluster`.
A typo (e.g. `events.Jet.ptX`) is caught: the processor runs a one-time expression
check on a small slice (with `custom/columns.py` derived fields attached) and
prints a warning listing any expressions that fail.

Expressions that produce missing values are safe: `None` entries (from
`ak.firsts` on empty events, `nearest` with no partner, …) are dropped
automatically during filling, with event weights kept aligned.

---
