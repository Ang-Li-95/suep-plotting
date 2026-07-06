# suep-plotting

A **config-driven, [coffea](https://github.com/CoffeaTeam/coffea)-based** plotting
framework for SUEP MDSNano analysis. Read CMS NanoAOD through coffea
**NanoEvents**, process it with a coffea **`ProcessorABC`** driven by
`processor.Runner`, apply scale factors with coffea's own correction tooling
(`correctionlib_wrapper` + `analysis_tools.Weights`), and produce
publication-quality CMS-style plots with
[mplhep](https://github.com/scikit-hep/mplhep).

**Key design goal:** adding a new histogram, selection, sample, or correction
should never require editing Python — just add a few lines to a YAML config and
re-run.

---

## Table of contents

1. [Repository layout](#repository-layout)
2. [Requirements](#requirements)
3. [Quick start](#quick-start)
4. [Configuration reference](#configuration-reference)
   - [samples.yaml](#samplesyaml)
   - [histograms.yaml](#histogramsyaml)
   - [selections.yaml](#selectionsyaml)
   - [corrections.yaml](#correctionsyaml)
   - [reweights.yaml](#reweightsyaml)
   - [derived_plots.yaml](#derived_plotsyaml)
5. [Expression language](#expression-language)
6. [Running locally](#running-locally)
7. [Running on Slurm](#running-on-slurm)
8. [How it works internally](#how-it-works-internally)
9. [Corrections & the scipy compat shim](#corrections--the-scipy-compat-shim)
10. [Recipes](#recipes)

---

## Repository layout

```
suep-plotting/
├── pyproject.toml                   # package metadata & dependencies
├── configs/
│   ├── samples.yaml                 # input file paths, cross sections, labels
│   ├── histograms.yaml              # histogram definitions (NanoEvents expressions)
│   ├── selections.yaml              # named event-/object-level cuts
│   ├── corrections.yaml             # correctionlib scale-factor definitions
│   ├── reweights.yaml               # event-/object-level reweighting (expressions & maps)
│   └── derived_plots.yaml           # profiles/projections/efficiency/ratio at plot time
├── custom/
│   └── columns.py                   # optional derive(events) -> events hook
└── src/suep_plot/
    ├── _compat.py                   # scipy shim so coffea.lookup_tools imports
    ├── processor.py                 # SuepProcessor(ProcessorABC) + Runner driver
    ├── histograms.py                # build hist.Hist from YAML, per-event & per-object fill
    ├── corrections.py               # correctionlib_wrapper + coffea Weights
    ├── reweights.py                 # expression/binned reweighting + suep-reweight maps
    ├── jme.py                       # jet energy corrections (JEC) + JER smearing
    ├── plot.py                      # mplhep CMS-style plotting (stack, overlay, data)
    ├── slurm.py                     # generates Slurm array job scripts
    ├── cli.py                       # entry points (run/plot/submit/reweight)
    └── cli_worker.py                # single-sample worker invoked by each Slurm task
```

---

## Requirements

The framework runs in the **`mds`** conda environment (coffea 2026.x, uproot 5,
awkward 2, hist 2, mplhep, correctionlib, numpy/scipy/matplotlib, pyyaml).

```bash
conda activate mds
```

| Package | Purpose |
|---------|---------|
| `coffea >= 2024` | NanoEvents, `ProcessorABC`/`Runner`, `analysis_tools.Weights`, `correctionlib_wrapper` |
| `uproot >= 5` / `awkward >= 2` | ROOT I/O and columnar jagged arrays (via coffea) |
| `hist >= 2.7` | Histogram objects with named axes |
| `mplhep >= 0.3` | CMS/ATLAS/LHCb plot styling |
| `correctionlib >= 2.3` | CMS-standard scale-factor evaluation |
| `scipy` | Clopper–Pearson intervals for efficiency plots |
| `pyyaml` | YAML config parsing |

Install once into the env (editable), which puts the console scripts
`suep-run` / `suep-plot` / `suep-submit` / `suep-reweight` on your `PATH`:

```bash
conda activate mds
cd suep-plotting
pip install -e .
```

The `-e` (editable) install points at this source tree, so code and config
edits take effect immediately, and `custom/columns.py` stays discoverable.
Uninstall anytime with `pip uninstall suep-plot`.

Without installing, the same commands work as
`PYTHONPATH=src python -m suep_plot.cli {run,plot,submit,reweight} …` from the
repo root.

Run the unit tests (fill logic, reweighting, plot helpers — no ROOT files
needed) with:

```bash
pip install -e ".[dev]"
pytest tests/
```

> **Note on corrections:** coffea's `lookup_tools` package (which contains
> `correctionlib_wrapper`) eagerly imports Rochester/double-Crystal-Ball modules
> that reference a scipy helper removed in newer scipy. `src/suep_plot/_compat.py`
> restores that helper *before* coffea loads, so no changes to the shared `mds`
> environment are required. See
> [Corrections & the scipy compat shim](#corrections--the-scipy-compat-shim).

---

## Quick start

### One command: process + plot

```bash
conda activate mds
cd suep-plotting
suep-run --plot
```

This reads every sample in `configs/samples.yaml`, fills every histogram in
`configs/histograms.yaml` (applying selections and corrections), writes **one
pickle file per sample** plus a cutflow table to `output/`, and renders all
figures to `output/plots/`.

Runs are **incremental**: a sample is skipped when its pickle is newer than
the configs, `custom/columns.py`, and its input files, so re-running after
adding one sample or histogram only processes what changed. Use `--force`
(`-f`) to reprocess everything (needed after *code* changes, which are not
tracked).

### 1. Process samples and fill histograms

```bash
suep-run -c configs -o output
```

```
output/
├── suep_mMed125_mDark2.pkl
├── qcd_ht300to500.pkl
└── ...
```

### 2. Plot

```bash
suep-plot output/ -o plots
```

The plotter loads all `.pkl` files, merges histograms across samples, and
writes one PNG + PDF per histogram (plus any `derived_plots.yaml` outputs,
`cutflow.txt`/`cutflow.csv`, and an **`index.html` gallery** for browsing
every figure from a single page). Figures render in parallel processes (`-j`,
auto by default). `-c` defaults to `./configs` when present, and **plot-time
styling keys** (`label`, `blind`, `rebin`, `flow`, `log_*`, …) are re-read
from `configs/histograms.yaml` on every invocation — so styling iterations
never touch the ROOT files:

```bash
suep-plot output/ -o plots --log --lumi 38.5
suep-plot output/ -o plots --normalize --formats png   # skip PDFs while iterating
suep-plot output/suep_mMed125_mDark2.pkl -o plots_signal_only
suep-plot output/ -o plots --save-root merged.root     # export for combine/ROOT
```

When data and stacked backgrounds are both present, 1D plots get a **Data/MC
ratio panel** with an MC-stat band (disable with `--no-ratio`).

`--lumi` does two things: it puts the luminosity in the CMS label **and**
normalizes every MC sample to `xs × lumi × 1000 / sumw` (so set real `xs`
values in `samples.yaml`). Without `--lumi`, plots show raw weighted event
counts.

### 3. Scale up with Slurm

```bash
suep-submit -c configs -o output --dry-run   # inspect
suep-submit -c configs -o output             # submit
bash output/slurm/merge_and_plot.sh          # after jobs finish
```

---

## Configuration reference

All behaviour is controlled by YAML files in `configs/`.

### samples.yaml

```yaml
suep_mMed125_mDark2:
  files:                                    # list of paths, globs, or xrootd URLs
    - "/path/to/NanoAOD/*.root"
  tree: Events                              # TTree name (default: Events)
  is_data: false                            # true for collision data
  xs: 1.0                                   # cross section in pb (MC; used by --lumi scaling)
  label: "SUEP $m_{Med}$=125"               # legend label (LaTeX ok)
  color: "tab:blue"                         # matplotlib color
  group: signal                             # signal / background (data via is_data)
  scale: 100                                # optional: draw signal x100 (legend shows "×100")
```

xrootd URLs (`root://…`) are passed through; paths with `*`/`?` are glob-expanded;
plain paths are used directly. `group` drives plot styling: `background` → stacked
fill, `signal` → step overlay (or fill if alone), `is_data: true` → error bars.
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

Plot-time fields (take effect on the next `suep-plot`, **no reprocessing**):

| Field | Default | Description |
|-------|---------|-------------|
| `label` | name | X-axis label (LaTeX between `$…$`). |
| `blind` | `false` | Don't draw data for this histogram (signal regions). |
| `rebin` | — | Merge N adjacent bins at plot time (1D only). |
| `flow` | — | `sum` folds under/overflow into the first/last bin. |
| `log_x` / `log_y` | `false` | Logarithmic axes for this histogram. |
| `log_z` | `false` | Log color scale (2D histograms). |

**2D histograms** use `expression_x`/`expression_y` with
`bins_x/lo_x/hi_x/bins_y/lo_y/hi_y` (or `edges_x`/`edges_y`) and
`label_x/label_y`. They render as colz and can feed the derived
profile/projection plots.

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
suep-run -c configs -o output                 # 1. fill histograms as usual
suep-reweight output/ --hist ht \
    --num data_2024 --den qcd \
    -o configs/ht_map.yaml                    # 2. map = shape ratio data/MC
# 3. reference it in configs/reweights.yaml (ht_dataMC above), then
suep-run -c configs -o output                 # 4. only affected samples re-run
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

## Running locally

```bash
# all samples (skips ones whose pickle is already up to date)
suep-run -c configs -o output --chunk-size 200000

# one sample
suep-run -c configs -o output -s suep_mMed125_mDark2

# multi-core (coffea FuturesExecutor)
suep-run -c configs -o output --workers 4

# reprocess everything (e.g. after editing suep_plot code)
suep-run -c configs -o output --force

# process + plot in one go
suep-run --plot --lumi 38.5 --log
```

Plotting is cheap and re-runnable (figures render in parallel; `-j 1` for
serial):

```bash
suep-plot output/ -o plots_log --log --lumi 38.5
suep-plot output/sig.pkl output/bkg.pkl -o cmp --log
suep-plot output/ -o plots --formats png -j 8       # fast iteration, PNG only
suep-plot output/ -o plots --save-root merged.root  # + ROOT export (post-scaling)
```

Each plot run also writes `cutflow.txt` / `cutflow.csv` (per-sample raw and
weighted event counts passing each selection, independently — handy for
acceptance checks and tables).

---

## Running on Slurm

One array task per sample; each writes a per-sample pickle. The generated scripts
export `PYTHONPATH=<repo>/src`, so no install is needed inside the job.

```bash
suep-submit \
    -c configs -o output --conda-env mds \
    --time 08:00:00 --mem 8000 --partition c --max-concurrent 50
# after jobs finish:
bash output/slurm/merge_and_plot.sh
```

| Flag | Default | Description |
|------|---------|-------------|
| `--conda-env` | `mds` | Environment activated inside each job. |
| `--time` / `--mem` | `04:00:00` / `8000` | Wall time / memory (MB) per job. |
| `--partition` / `--account` | — | Slurm partition / account. |
| `--chunk-size` | `100000` | Events per chunk. |
| `--workers` | `1` | Worker processes per job (also sets `--cpus-per-task`). |
| `--max-concurrent` | — | Cap on simultaneous array tasks (`%N`). |
| `--dry-run` | off | Generate scripts without submitting. |

Resubmit failures by task index (line number − 1 in `slurm/sample_list.txt`):
`sbatch --array=3,7 output/slurm/job.sh`. Slurm jobs always reprocess their
sample (`--force`); the incremental skip only applies to local `suep-run`.
`merge_and_plot.sh` forwards extra arguments to `suep-plot`
(e.g. `bash output/slurm/merge_and_plot.sh --lumi 38.5 --log`).

---

## How it works internally

```
configs/*.yaml
      │
      ▼
  processor.run_all()
   1. Load YAML configs
   2. build_correctors()  →  correctionlib_wrapper per correction
   3. SuepProcessor(ProcessorABC) holds the configs
   4. For each sample (skipped when its pickle is up to date):
        Runner({sample: files}, proc, treename)
        NanoEvents(chunk) → process(events):
          • Weights(n); add genWeight; apply_corrections();
            apply_reweights() → Weights
          • evaluate selections → event/object masks
          • fill hist.Hist (per-event & per-object); expression results
            are cached per chunk, so shared expressions evaluate once
          • cutflow: raw + weighted counts per selection
        → accumulate across chunks
   5. Save output/<sample>.pkl  {histograms, samples, hist_defs, sumw,
                                 nevents, cutflow}
      │
      ▼
  plot.plot_all()  → merge per-sample pickles → xs×lumi scaling (if --lumi)
                   → cutflow.txt/csv → optional ROOT export (--save-root)
                   → CMS-style figures in parallel (+ derived plots)
```

- **Reading:** `processor.Runner` + `NanoAODSchema` handle file opening, chunking
  (`chunksize`), schema application, and accumulation. `IterativeExecutor` by
  default; `FuturesExecutor` with `--workers > 1`.
- **Histogram storage:** each `hist.Hist` has a `dataset` `StrCategory` axis
  (`growth=True`) plus the physics axis/axes, with `Weight()` storage (value +
  sumw²).
- **Plotting:** samples are grouped — `background` stacked fill, `signal` step
  overlay (fill if alone; `--normalize` scales to unit area), data as error bars.
  The CMS label reads "Simulation Preliminary" for MC-only, "Preliminary" + lumi
  when data is present.

---

## Corrections & the scipy compat shim

Corrections use **coffea's own tools**:

- `coffea.lookup_tools.correctionlib_wrapper` wraps each correctionlib
  `Correction` so it evaluates on (flattened) awkward arrays;
- `coffea.analysis_tools.Weights` accumulates the event weight (genWeight × every
  applicable correction) and is ready for up/down systematic variations.

`coffea.lookup_tools.__init__` eagerly imports Rochester / double-Crystal-Ball
modules that do `from scipy.stats._continuous_distns import _lazywhere` — a helper
removed in newer scipy (the `mds` env has scipy 1.18). That import failure would
otherwise block `correctionlib_wrapper`, `jetmet_tools`, and `extractor`.
`src/suep_plot/_compat.py` re-installs the small `_lazywhere` function **before**
coffea loads (imported at the top of `suep_plot/__init__.py`), which restores
coffea's documented fallback path. It's a no-op once scipy (or a future coffea
release) provides the symbol, and it changes nothing in the shared environment.

---

## Recipes

**Add a histogram** — append to `histograms.yaml`:

```yaml
leading_jet_pt:
  expression: "ak.firsts(events.Jet.pt)"
  bins: 50
  lo: 0
  hi: 500
  label: "Leading jet $p_T$ [GeV]"

muon_dxy:
  expression: "events.Muon.dxy"
  bins: 100
  lo: -0.05
  hi: 0.05
  label: "$d_{xy}^{\\mu}$ [cm]"
  per_object: true
```

**Add a sample** — append to `samples.yaml` (see [samples.yaml](#samplesyaml)).

**Add a selection** — append to `selections.yaml`:

```yaml
dimuon:
  label: "$N_{\\mu} \\geq 2$"
  expression: "ak.num(events.Muon) >= 2"

large_csc_cluster:
  label: "CSC cluster size > 100"
  expression: "ak.any(events.cscMDSHLTCluster.size > 100, axis=1)"
```

**Enable pileup reweighting / muon SFs** — uncomment the examples in
`corrections.yaml`. `object_sf` evaluates per object and multiplies the per-event
product into the weight.

**Reweight events or objects for a study** — see
[reweights.yaml](#reweightsyaml): expression- or map-based, event- or
object-level, with `suep-reweight` to derive data/MC maps from processed
histograms.

**Normalize MC to a luminosity** — set real `xs` values (pb) in `samples.yaml`
and pass `--lumi` (fb⁻¹) at plot time; each MC sample is scaled by
`xs × lumi × 1000 / sumw`:

```bash
suep-plot output/ -o plots --lumi 38.5
```

**Variable binning** — use `edges` instead of `bins/lo/hi`:

```yaml
ht_coarse:
  expression: "ak.sum(events.Jet.pt, axis=1)"
  edges: [0, 100, 200, 400, 800, 1600]
  label: "$H_T$ [GeV]"
  flow: sum          # fold overflow into the last bin
```

**Blind a signal region** — add `blind: true` to the histogram; data is not
drawn (MC and signal still are). Takes effect at plot time, no reprocessing.

**Overlay a small signal on large backgrounds** — set `scale: 100` on the
signal sample; the legend shows "×100" automatically.

**Cutflow table** — written automatically to `<plots>/cutflow.txt` and
`.csv` on every `suep-plot`, one row per selection per sample (raw and
weighted counts).

**Export histograms to ROOT** (for `combine`, `TH1`-based tooling):

```bash
suep-plot output/ -o plots --lumi 38.5 --save-root merged.root
# layout: <histogram>/<sample>, e.g. met_pt/qcd_ht300to500 (TH1D with Sumw2)
```

**Jet energy corrections + resolution smearing (JEC/JER)** — these rescale the
jet four-momentum, so they run in `custom/columns.py` (before selections and
fills), *not* as a weight in `corrections.yaml`:

```python
from suep_plot.jme import correct_jets

def derive(events):
    events = correct_jets(events)        # 2024 Summer24 MC defaults
    return events
```

Afterwards every `events.Jet` expression (HT, jet pT, `good_jets`, …) uses
corrected jets. `suep_plot/jme.py` follows the JME prescription: undo
`rawFactor`, apply the compound `L1L2L3Res` JEC from jsonpog
`jet_jerc.json.gz`, then (MC) smear with the official `JERSmear` helper —
gen-matched scaling within `dR < 0.2` and `3σ`, deterministic stochastic
smearing (seeded from the event number) otherwise — and re-sort jets by the
new pT. Systematics are one argument away:

```python
events = correct_jets(events, variation="jec_up")    # jec_down / jer_up / jer_down
```

For data pass the run-specific tag and disable smearing:
`correct_jets(events, jec_tag="Summer24Prompt24_RunX_V1_DATA", smear=False)`.
Payload files resolve from `$CORRECTIONLIB_DATA`, then cvmfs
jsonpog-integration. Note: MET is not propagated, and `pt`/`mass` are
replaced in place (re-run with the example removed to get uncorrected jets —
processing is incremental, but `custom/columns.py` edits are tracked, so
affected samples re-run automatically).

**Custom derived columns** — edit `custom/columns.py`; `derive(events)` returns the
(augmented) events array. Attach fields with `ak.with_field(events, value, "name")`
and reference them as `events.name` in any expression.

**xrootd files** — list `root://host//store/…` URLs under `files:`; coffea/uproot
handle them natively (ensure a valid grid proxy / kerberos token).
