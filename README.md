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
│   └── derived_plots.yaml           # profiles/projections/efficiency/ratio at plot time
├── custom/
│   └── columns.py                   # optional derive(events) -> events hook
└── src/suep_plot/
    ├── _compat.py                   # scipy shim so coffea.lookup_tools imports
    ├── processor.py                 # SuepProcessor(ProcessorABC) + Runner driver
    ├── histograms.py                # build hist.Hist from YAML, per-event & per-object fill
    ├── corrections.py               # correctionlib_wrapper + coffea Weights
    ├── plot.py                      # mplhep CMS-style plotting (stack, overlay, data)
    ├── slurm.py                     # generates Slurm array job scripts
    ├── cli.py                       # command-line entry points (run/plot/submit)
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

Install as a package (optional — the console scripts `suep-run` / `suep-plot` /
`suep-submit` become available):

```bash
pip install -e .
```

Or run without installing, straight from the repo:

```bash
cd suep-plotting
PYTHONPATH=src python -m suep_plot.cli run -c configs -o output
```

> **Note on corrections:** coffea's `lookup_tools` package (which contains
> `correctionlib_wrapper`) eagerly imports Rochester/double-Crystal-Ball modules
> that reference a scipy helper removed in newer scipy. `src/suep_plot/_compat.py`
> restores that helper *before* coffea loads, so no changes to the shared `mds`
> environment are required. See
> [Corrections & the scipy compat shim](#corrections--the-scipy-compat-shim).

---

## Quick start

### 1. Process samples and fill histograms

```bash
conda activate mds
cd suep-plotting
PYTHONPATH=src python -m suep_plot.cli run -c configs -o output
```

This reads every sample in `configs/samples.yaml`, fills every histogram in
`configs/histograms.yaml` (applying selections and corrections), and writes
**one pickle file per sample** to `output/`:

```
output/
├── suep_mMed125_mDark2.pkl
├── qcd_ht300to500.pkl
└── ...
```

### 2. Plot

```bash
PYTHONPATH=src python -m suep_plot.cli plot output/ -o plots -c configs
```

The plotter loads all `.pkl` files, merges histograms across samples, and writes
one PNG + PDF per histogram (plus any `derived_plots.yaml` outputs). It never
touches the ROOT files, so re-run it freely with different styles:

```bash
PYTHONPATH=src python -m suep_plot.cli plot output/ -o plots --log --lumi 38.5
PYTHONPATH=src python -m suep_plot.cli plot output/ -o plots --normalize
PYTHONPATH=src python -m suep_plot.cli plot output/suep_mMed125_mDark2.pkl -o plots_signal_only
```

### 3. Scale up with Slurm

```bash
PYTHONPATH=src python -m suep_plot.cli submit -c configs -o output --dry-run   # inspect
PYTHONPATH=src python -m suep_plot.cli submit -c configs -o output             # submit
bash output/slurm/merge_and_plot.sh                                           # after jobs finish
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
  xs: 1.0                                   # cross section in pb (MC; stored for future scaling)
  label: "SUEP $m_{Med}$=125"               # legend label (LaTeX ok)
  color: "tab:blue"                         # matplotlib color
  group: signal                             # signal / background (data via is_data)
```

xrootd URLs (`root://…`) are passed through; paths with `*`/`?` are glob-expanded;
plain paths are used directly. `group` drives plot styling: `background` → stacked
fill, `signal` → step overlay (or fill if alone), `is_data: true` → error bars.
`sumw` (Σ genWeight) and `nevents` are recorded per sample in the pickle for
future cross-section × luminosity scaling.

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

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `expression` | yes | — | NanoEvents expression (see [Expression language](#expression-language)). |
| `bins` / `lo` / `hi` | yes | — | Axis binning. |
| `label` | no | name | X-axis label (LaTeX between `$…$`). |
| `per_object` | no | `false` | If `true`, expression returns a jagged array (e.g. one value per jet); it is flattened, with per-event weights repeated per object. Otherwise it must return one value per event. |
| `selections` | no | `[]` | Selection names from `selections.yaml`, AND-ed. |
| `weight` | no | — | Extra per-event weight (× genWeight × corrections). |

**2D histograms** use `expression_x`/`expression_y` with
`bins_x/lo_x/hi_x/bins_y/lo_y/hi_y` and `label_x/label_y`. They render as colz and
can feed the derived profile/projection plots.

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

### derived_plots.yaml

Computed at **plot time** from saved histograms — no reprocessing. Types:
`profile_x`/`profile_y` (from a 2D hist), `projection_x`/`projection_y`,
`efficiency` (Clopper–Pearson errors), `ratio`. See the file header for fields.

---

## Expression language

Expressions in `histograms.yaml`, `selections.yaml`, `corrections.yaml`, and any
`weight` are `eval`'d with these names in scope:

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
check on a small slice and prints a warning listing any expressions that fail.

---

## Running locally

```bash
# all samples
PYTHONPATH=src python -m suep_plot.cli run -c configs -o output --chunk-size 200000

# one sample
PYTHONPATH=src python -m suep_plot.cli run -c configs -o output -s suep_mMed125_mDark2

# multi-core (coffea FuturesExecutor)
PYTHONPATH=src python -m suep_plot.cli run -c configs -o output --workers 4
```

Plotting is cheap and re-runnable:

```bash
PYTHONPATH=src python -m suep_plot.cli plot output/ -o plots_log --log --lumi 38.5
PYTHONPATH=src python -m suep_plot.cli plot output/sig.pkl output/bkg.pkl -o cmp --log
```

---

## Running on Slurm

One array task per sample; each writes a per-sample pickle. The generated scripts
export `PYTHONPATH=<repo>/src`, so no install is needed inside the job.

```bash
PYTHONPATH=src python -m suep_plot.cli submit \
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
| `--max-concurrent` | — | Cap on simultaneous array tasks (`%N`). |
| `--dry-run` | off | Generate scripts without submitting. |

Resubmit failures by task index (line number − 1 in `slurm/sample_list.txt`):
`sbatch --array=3,7 output/slurm/job.sh`.

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
   4. For each sample:  Runner({sample: files}, proc, treename)
        NanoEvents(chunk) → process(events):
          • Weights(n); add genWeight; apply_corrections() → Weights
          • evaluate selections → event/object masks
          • fill hist.Hist (per-event & per-object)
        → accumulate across chunks
   5. Save output/<sample>.pkl  {histograms, samples, hist_defs, sumw, nevents}
      │
      ▼
  plot.plot_all()  → merge per-sample pickles → CMS-style PNG/PDF (+ derived plots)
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

**Custom derived columns** — edit `custom/columns.py`; `derive(events)` returns the
(augmented) events array. Attach fields with `ak.with_field(events, value, "name")`
and reference them as `events.name` in any expression.

**xrootd files** — list `root://host//store/…` URLs under `files:`; coffea/uproot
handle them natively (ensure a valid grid proxy / kerberos token).
