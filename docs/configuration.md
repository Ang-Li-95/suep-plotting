# Configuration reference

A **config set** is a directory of YAML files describing one study. `suep-run`,
`suep-submit` and `suep-plot` take it with `-c`:

```
configs/configs_mds_signal/
├── set.yaml             # optional: parent set, one-line description, abstract
├── samples.yaml         # which datasets to run (from ../../datasets.yaml)
├── histograms.yaml      # what to fill: regions -> cuts -> variables
├── selections.yaml      # named event- and object-level cuts
├── corrections.yaml     # correctionlib scale factors
├── reweights.yaml       # event-/object-level reweighting
├── derived_plots.yaml   # efficiencies, ratios, profiles ... at plot time
├── columns.yaml         # derive() parameters and steps -- see derived-columns.md
└── variables.yaml       # optional: this set's overrides of the variable catalog
```

Every file is read by `src/suep_plot/config.py`, so the directives below mean
the same thing in all of them. To see what a set really defines once
everything is inherited and expanded:

```bash
suep-config show configs/configs_mds_signal --grep cluster_size
suep-config check configs/configs_mds_signal      # references, then every expression
```

---

## Config sets and inheritance

### `set.yaml`

```yaml
extends: ../configs_mds          # every file this set does not have comes from there
description: "configs_mds with the standard MDS clustering: min cluster size 50"
abstract: true                   # bases only: never run on their own
```

- **`extends`** -- any config file missing from the set behaves as
  `_extends: <parent>`. A set that changes one setting is therefore two files:
  `set.yaml` and the file that differs (see `configs/configs_mds_minpts50/`).
  A file the set *does* have keeps its own meaning: start it with
  `_extends: <parent>` to change only part of the parent's file, or it
  replaces the parent's file entirely (`suep-config check` warns about this).
- **`description`** -- listed when `-c` names something that is not a set.
- **`abstract`** -- `common`, `common_gen` and `common_clustersrc` are bases
  that only exist to be extended; `suep-run` / `suep-submit` refuse them.

### The bases

```
configs/common/            reco histograms + selections + derived plots,
  │                        rpc_mode: match, the four reco steps
  ├── configs_mds/                  _extends ../common      (and nothing else)
  │     └── configs_mds_minpts50/   set.yaml + columns.yaml (min cluster size 50)
  ├── configs_mds_data/             + the isolation cuts
  ├── configs_mds_cosmics/          the cluster plots that need no prompt object
  ├── common_clustersrc/            every cluster plot cut to one population
  │     └── configs_mds_src_*/        the six sets define the population
  └── common_gen/          + LLP collection, truth cuts, efficiency chain,
        │                    and the four gen steps
        ├── configs_mds_signal/     _extends ../common_gen  (and nothing else)
        ├── configs_mds_gen/        + the LLP shower-shape family
        └── configs_mds_rpcmerge/   configs_mds_signal with rpc_mode: merge
```

`common` carries only the **reco** steps (`jerc, jet_id, clusters,
cluster_isolation`); `common_gen` adds the four gen ones. A study with no
gen-level fills does not pay for them, and `events.llp` being absent is what
makes a stray gen expression fail at startup instead of filling empty.

---

## Directives

Five keys are understood in **every** config file. They are directives, not
definitions: like any `_`-prefixed key they never become a histogram, a
selection or a sample. An unknown `_` key (a typo like `_extend:`) is an
error, as is a file that is not valid YAML.

### `_extends` -- inherit the same file from another set

```yaml
_extends: ../configs_mds_signal

some_selection: null            # drop an inherited definition
regions:
  csc:
    objects:
      iso_muon: {cut: csc_cluster_iso_muon, vars: [csc_cluster_size]}   # add a cut
```

The same-named file from the other set is loaded first and this file's keys are
merged over it: **mappings merge deeply, lists are replaced, `null` deletes**.
Chains are followed, a list of parents is merged left to right, and a cycle is
an error.

### `_repeat` and `_axes` -- write a per-system family once

```yaml
_axes:
  systems:
    - {SYS: csc, SYSU: CSC}
    - {SYS: dt,  SYSU: DT}

_repeat:
  - over: systems                 # or an inline list of mappings
    defs:
      <SYS>_cluster_matched: "events.<SYS>Cluster.matched"
```

- The sigil is `<NAME>`, **not** `{NAME}` -- the labels are full of LaTeX braces.
- Substitution reaches keys and every string leaf, including inside lists.
- A value that is *exactly* one placeholder keeps the substituted **type**, so
  `hi: <HI>` with `HI: 500` stays the integer 500.
- Axes are inherited through `_extends`; a `regions:` block may hold its own
  `_repeat` (see below).
- A generated name colliding with an explicit definition is an error.

### `_include` -- merge in a shared fragment

```yaml
_include: [../_shared/cuts.yaml]
```

Definitions from the named files are merged in, then this file's keys over them.
Unlike `_extends`, which follows the config *set*, an include names one file.
No shipped set uses it.

### `_registry` -- select from a pool

What `samples.yaml` does with the central dataset registry:

```yaml
_registry: ../../datasets.yaml

suep_mDark2_temp1:                     # take the registry entry verbatim
suep_mDark2_temp2: {color: "tab:red"}  # ... with these keys overridden
signal_alias:
  _from: suep_mDark2_temp1             # base it on a differently-named entry
  label: "the same data, another name"
```

`_registry` and `_include` mean opposite things for an entry with no value: an
include *merges*, so `name:` deletes; a registry is a pool you *select* from, so
`name:` takes.

The up-to-date check of `suep-run` follows all of these (and `set.yaml`), so
editing an inherited or included file reprocesses every set that reads it.

---

## The variable catalog: `variables.yaml`

Every variable a histogram can plot, written once at the repo root:

```yaml
csc_cluster_size: {expression: events.cscCluster.size, bins: [50, 0, 500], label: 'CSC cluster $N_{\mathrm{hits}}$', log_y: true}
llp_decay_r:      {expression: events.llp.decayR, bins: [60, 0, 1200], label: 'LLP decay $r$ [cm]'}
llp_pt:           {expression: events.llp.pt, edges: [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50], label: 'LLP $p_T$ [GeV]'}
```

| Key | Meaning |
|---|---|
| `expression` | what is filled (see [Expression language](#expression-language)) |
| `bins: [n, lo, hi]` or `edges: [...]` | the default binning |
| `label` | the axis title |
| `log_x`, `log_y`, `flow`, `rebin`, `blind` | optional display defaults of its 1D histograms |

Anything else is an error. A histogram names the variable and overrides any of
these for itself. A config set can retune a default for all of its histograms
in its own `configs/<set>/variables.yaml`, merged over the global file (and
inherited through `_extends` / `set.yaml` like every other file):

```yaml
# configs/configs_xyz/variables.yaml
llp_decay_r: {bins: [90, 0, 900]}       # every histogram of llp_decay_r follows
my_new_variable: {expression: "events.llp.pt / events.llp.energy", bins: [50, 0, 1], label: '$p_T/E$'}
```

---

## `histograms.yaml`

### Regions: event cuts → object cuts → variables

Most histograms are written as a **region tree**:

```yaml
regions:
  csc:                                      # region
    label: CSC clusters                     # shown on the figures (default: the name)
    events: [has_csc_cluster]               # event-level cuts, AND-ed (may be omitted)
    per_event: [csc_cluster_n]              # filled once per event after the event cuts
    objects:                                # object-level cuts -> variables
      all:                                  # no cut: every cluster
        vars: &csc_vars
          - csc_cluster_size
          - csc_cluster_eta
          - {var: csc_cluster_rpc_bx, selections: [csc_cluster_has_rpc]}
          - {name: csc_cluster_dr_muon_vs_jet, x: csc_cluster_dr_muon, y: csc_cluster_dr_jet}
      iso_muon:
        cut: csc_cluster_iso_muon           # one selection, or a list (AND-ed)
        label: muon-isolated
        vars: *csc_vars                     # YAML anchors reuse a list in one file
```

Each entry becomes one histogram named **`<region>/<cut>/<variable>`**
(`csc/iso_muon/csc_cluster_size`), or `<region>/<variable>` for `per_event`.
The name is also the figure's folder: `plots/csc/iso_muon/csc_cluster_size.png`.

- **Selections** of a histogram: the region's `events`, then the cut's `cut`,
  then the entry's own `selections` (all AND-ed).
- **Per object or per event** is read off the expression: a variable with one
  entry per object (`events.cscCluster.size`) fills once per object passing the
  cuts; a per-event variable under an object cut fills once per event with at
  least one passing object.
- **Labels**: the x-axis title is the variable's; the region and cut labels go
  on the figure ("CSC clusters · muon-isolated"). A cut's label defaults to its
  name, except `all`.

An **entry** is one of:

| Entry | Histogram |
|---|---|
| `csc_cluster_size` | the catalog variable, 1D, leaf name = the variable |
| `[llp_decay_absz, llp_decay_r]` | 2D, x and y from the catalog, leaf `llp_decay_absz_vs_llp_decay_r` |
| `{var: llp_pt, bins: [50, 0, 50], log_y: true}` | 1D with overrides |
| `{name: llp_decay_rz, x: llp_decay_absz, y: {var: llp_decay_r, bins: [60, 0, 900]}}` | 2D with a name and a per-axis override |
| `{var: x, selections: [...], weight: ...}` | extra cuts or a weight for this histogram only |

Overrides are any histogram key (binning, `label`, `weight`, display keys);
the same variable listed twice under one cut needs a `name:` for one of them.

**Inheritance.** Regions merge deeply, so a child set adds a cut to an
inherited region with one entry, or deletes one with `<cut>: null`. A `vars`
list is replaced, not appended: to change one variable of an inherited cut,
restate its list. `per_event: []` clears an inherited list.

A per-system family of regions can be written once with a `_repeat` inside
`regions:` (using the file's `_axes`).

### Histograms written out on their own

Outside `regions:` a histogram is defined directly; this is also the form the
regions expand to, and what `suep-config show` displays:

```yaml
leading_jet_pt:
  x: jet_pt                       # a catalog variable ...
  expression: "ak.firsts(events.Jet.pt)"     # ... with its expression replaced
  bins: [50, 0, 500]

met_vs_ht:                        # 2D: an x and a y axis
  x: {expression: "events.PuppiMET.pt", bins: [40, 0, 400], label: 'MET [GeV]'}
  y: {var: ht}
  log_z: true

old_style:                        # the long form still works
  expression: "events.Muon.dxy"
  bins: 100
  lo: -0.05
  hi: 0.05
  label: '$d_{xy}^{\mu}$ [cm]'
```

Shorthands: `bins: [n, lo, hi]` for a regular axis (and `bins_x`, `bins_y`).
An `x:` / `y:` value is a variable name, `{var: name, <overrides>}`, or an
inline `{expression, bins | edges, label}`. In 1D, keys written at the top
level override the x axis. Replacing inherited `edges` with `bins` (or the other
way round) needs `edges: null` (`bins: null`) for the old one.

Fill-time keys (changing them needs `suep-run` again): `expression` /
`expression_x` / `expression_y`, the binning, `selections`, `weight`.

Plot-time keys (take effect on the next `suep-plot`, **no reprocessing**):

| Key | Default | Effect |
|---|---|---|
| `label` (`label_x`, `label_y`) | name | axis titles |
| `annotation` | region · cut | text on the figure |
| `blind` | `false` | don't draw data |
| `rebin` | -- | merge N adjacent bins (1D) |
| `flow` | `sum` | fold under/overflow into the edge bins; `none` drops it |
| `log_x` / `log_y` / `log_z` | `false` | log axes / colour scale |

`per_object` is no longer written: it is read off the expression (jagged =
per object). An explicit value must agree with the expression, or the run
stops naming the histogram.

#### `variants` (older form)

`variants:` and `_variant_sets` still work -- a histogram repeated under extra
selections as `<name>_<suffix>` -- but the shipped configs use regions instead,
where the same split is a cut under a region.

---

## `selections.yaml`

Named boolean cuts. A selection is just its expression, or a mapping when it
needs a `label` or a `collection`:

```yaml
has_csc_cluster: "ak.num(events.cscCluster) >= 1"      # one entry per event

csc_cluster_iso_muon: "events.cscCluster.drMuon >= 0.4" # one entry per cluster

muon_sel_foriso:                                        # read by the isolation step
  collection: Muon
  label: 'loose muon, $p_T > 10$ GeV'
  expression: "(events.Muon.pt > 10) & (abs(events.Muon.eta) < 2.4) & events.Muon.looseId"
```

Whether a selection is event- or object-level is read off its mask (one entry
per object = object-level); a `level:` key is accepted but must agree.
Object-level selections filter the objects a per-object histogram fills;
applied to a per-event histogram they mean "at least one object passes".

### Selections as columns

Every selection is also attached to `events` as a **column of its own name**,
so any expression can reach the mask instead of restating the cut:

```yaml
# variables.yaml
n_electron_sel: {expression: "ak.sum(events.electron_sel, axis=1)", bins: [6, -0.5, 5.5], label: '$N_{e}$'}
```

| Want | Expression |
|---|---|
| how many objects pass | `ak.sum(events.electron_sel, axis=1)` |
| does any object pass | `ak.any(events.electron_sel, axis=1)` |
| an event-level mask | `events.high_met` -- already flat |

- Attached **after** the `custom/` derived columns, so a selection over a
  derived collection (`events.cscCluster.matched`) becomes a column too.
- Missing values become `False`.
- Selections may read each other's columns in any order.
- A selection whose name is already a field of `events` is skipped with a
  message: a selection named `Jet` must not redefine the collection.

---

## `samples.yaml` and `datasets.yaml`

Datasets are defined once in [`datasets.yaml`](../datasets.yaml); each set's
`samples.yaml` selects from it with `_registry` (see above).

```yaml
dy_2mu_50to120:
  files:                       # files, globs or directories, local or xrootd
    - root://eos.grid.vbc.ac.at//eos/vbc/experiments/cms/store/user/.../MDSnano/...
  tree: Events                 # default: Events
  is_data: false
  xs: 1.0                      # pb, for --lumi; 1.0 here is a placeholder
  label: 'DY$\to2\mu$ (50-120)'
  color: tab:orange
  linestyle: "--"              # optional
  group: background            # signal / background (data via is_data)
  scale: 100                   # optional: draw signal x100
  max_files: 1000              # optional: the first N files only
  files_per_job: 25            # optional: suep-submit's shard size for it
```

A directory is walked recursively for `*.root` (over xrootd too). `suep-submit`
freezes the resolved file list per task at submission time. `group` drives the
styling: `background` stacked, `signal` overlaid, `is_data` points. Every MC
`xs` in the MDS sets is the placeholder `1.0`, so never pass `--lumi` to them
(`suep-plot` warns).

---

## `corrections.yaml`

Scale factors through coffea's tooling. An empty file means no corrections;
data samples are always skipped. A correction that fails to load stops the run.

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
| `kind` | no | `event_weight` (default; one weight per event) or `object_sf` (one SF per object; per-event weight = product over objects). |
| `apply_to` | no | `all` (default) or a list of sample groups. |

---

## `reweights.yaml`

Analysis-level **reweighting** -- flatten a spectrum, match MC kinematics to
data, emulate a trigger turn-on. Weights multiply into the event weight
alongside genWeight and the scale factors, so they also show up in cutflows
and `--lumi` scaling. Config errors here are fatal.

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
| `expression` | -- | NanoEvents expression → weight (event-level scalar, or jagged with `level: object`). |
| `variable` (+ `edges`, `weights`) | -- | 1D binned lookup. 2D: `variable_x/variable_y`, `edges_x/edges_y`, nested `weights`. |
| `file` | -- | Load map fields from another YAML, relative to the reweights.yaml that names it; entry keys override. |
| `level` | `event` | `object`: one weight per object, event weight = product over objects. |
| `apply_to` | `all` | Sample groups to reweight (MC only). |
| `samples` | -- | Explicit sample list (overrides `apply_to`; may include data). |
| `clamp` | `false` | Binned maps: out-of-range values use the nearest bin (default: weight 1 outside). |
| `fill_only` | `false` | Don't apply globally; only where a histogram sets `weight: "@<name>"`. |

A histogram's `weight` may be a jagged expression or a `"@<map name>"`
reference: for per-object histograms each object carries its own weight, for
per-event histograms the per-object weights are multiplied. Mark such maps
`fill_only: true` or the weight is applied twice (the run warns).

Deriving a map from processed samples:

```bash
suep-run -c configs/configs_mds -o output                  # 1. fill as usual
suep-reweight output/ --hist jet/ht --num data_2024 --den qcd \
    -o configs/configs_mds/ht_map.yaml                     # 2. shape ratio data/MC
# 3. reference it in reweights.yaml (ht_dataMC above), then run again
```

---

## `derived_plots.yaml`

Computed at **plot time** from saved histograms -- no reprocessing.

| `type` | Reads | Draws |
|---|---|---|
| `efficiency` | `numerator`, `denominator` | ratio with Clopper-Pearson intervals; `spans: [{lo, hi, label}]` shades x ranges; 2D sources give a 2D map |
| `ratio` | `numerator`, `denominator` | plain ratio |
| `profile_x` / `profile_y` | `source` (2D) | mean of y per x bin (or x per y) |
| `projection_x` / `projection_y` | `source` (2D) | projection onto one axis |
| `sig_vs_bkg` | `signal`, `background` | signal samples from one histogram, background samples from another, unit-normalized (`normalize: false` keeps yields) |

Common keys: `label_x` (default: the source's axis title), `label_y`, `log_y`.

**Region-aware form** -- one entry for a list of variables:

```yaml
matchfrac_csc:
  type: efficiency
  region: csc
  numerator: matched              # cuts of that region
  denominator: all
  label_y: fraction of CSC clusters LLP-matched
  vars: [csc_cluster_size, csc_cluster_eta, csc_cluster_time]
```

expands to `csc/matchfrac_csc/csc_cluster_size` (numerator
`csc/matched/csc_cluster_size`, denominator `csc/all/csc_cluster_size`) and so
on. Single-source types take `cut:` instead; a cut left out means the region's
per-event histogram. The explicit form names the histograms itself:

```yaml
eff_trigger:
  type: efficiency
  numerator: hlt_csccluster_loose/evt_max_csc_cluster_size
  denominator: csc/evt_max_csc_cluster_size
```

---

## Expression language

Expressions in histograms, variables, selections, corrections, reweights and
any `weight` are evaluated with these names in scope:

- `events` -- the coffea NanoEvents array (also `ev`)
- `ak` -- awkward, `np` -- numpy
- builtins `abs`, `len`, `min`, `max`

```python
events.Jet.pt                                  # jagged per-jet pt
ak.num(events.Muon)                            # muons per event
ak.sum(events.Jet.pt, axis=1)                  # scalar HT
abs(events.Jet.eta) < 2.4                      # object-level mask
ak.firsts(events.Jet.pt)                       # None-safe leading jet
events.Jet.nearest(events.Muon).delta_r(events.Jet)   # dR to nearest muon
```

Missing values (`ak.firsts` on an empty event, `nearest` with no partner) are
dropped while filling, with the weights kept aligned. To see which collections
and fields a sample has after `derive()`:

```bash
suep-config fields configs/configs_mds -s suep_mDark2_temp1 --grep cscCluster --ranges
```

## When something is wrong

Nothing fails quietly:

- **Config mistakes** stop the run before any file is opened, naming the file:
  invalid YAML, an unknown directive, an unknown variable / selection /
  reweight map (with the closest name suggested), an unknown key in a region,
  axis or catalog entry, a histogram without an expression or binning, a bad
  `columns.yaml`, a step without what it needs.
- **Expressions** are evaluated on the first events of every sample's first
  file before processing starts; any that fails stops the run with a list. A
  failure later, on some chunk, stops it too, naming the histogram or
  selection, the sample and the expression. A histogram that cannot fill on
  one of the set's samples (a branch that sample lacks) therefore belongs in a
  set without that sample.
- `suep-config check <set>` runs all of this without processing anything.
