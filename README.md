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
3. [Setup](#setup) — [install](#1-install-the-package), [grid proxy](#2-grid-proxy-for-xrootd-inputs), [tests](#3-check-the-install)
4. [Quick start](#quick-start)
5. [The commands](#the-commands) — `suep-run`, `suep-plot`, `suep-submit`, `suep-status`, `suep-reweight`
6. [Config sets in this repo](#config-sets-in-this-repo)
7. [Derived-column settings](#derived-column-settings-columnsyaml)
8. [Configuration reference](#configuration-reference)
   - [samples.yaml](#samplesyaml)
   - [histograms.yaml](#histogramsyaml)
   - [selections.yaml](#selectionsyaml)
   - [corrections.yaml](#correctionsyaml)
   - [reweights.yaml](#reweightsyaml)
   - [derived_plots.yaml](#derived_plotsyaml)
9. [Expression language](#expression-language)
10. [Running locally](#running-locally)
11. [Running on Slurm](#running-on-slurm)
12. [Helper scripts](#helper-scripts)
13. [How it works internally](#how-it-works-internally)
14. [Corrections & the scipy compat shim](#corrections--the-scipy-compat-shim)
15. [Recipes](#recipes)
16. [Troubleshooting](#troubleshooting)

---

## Repository layout

```
suep-plotting/
├── pyproject.toml                   # package metadata, dependencies, console scripts
├── datasets.yaml                    # central dataset registry (paths, xs, labels)
├── configs/                         # one subdirectory per config set
│   ├── configs_mds/                 # a "config set" = these files
│   │   ├── samples.yaml             # which datasets to run, cross sections, styling
│   │   ├── histograms.yaml          # histogram definitions (NanoEvents expressions)
│   │   ├── selections.yaml          # named event-/object-level cuts
│   │   ├── corrections.yaml         # correctionlib scale-factor definitions
│   │   ├── reweights.yaml           # event-/object-level reweighting (expressions & maps)
│   │   ├── derived_plots.yaml       # profiles/projections/efficiency/ratio at plot time
│   │   └── columns.yaml             # optional: parameters + enabled steps of derive()
│   └── configs_mds_grid/  configs_mds_gen/  configs_mds_data/  …
│                                    # the other config sets (see "Config sets")
├── custom/                          # derive(events) -> events hook: LLP + DBSCAN
│   ├── columns.py                   #   derive() itself + the step pipeline
│   ├── params.py                    #   constants, defaults, columns.yaml reader
│   ├── clustering.py                #   DBSCAN of one rechit system
│   ├── llp.py                       #   events.llp + per-LLP rechit spread
│   └── isolation.py                 #   cluster dR to the nearest prompt object
├── scripts/                         # standalone plotting/inspection tools
│   ├── compare_eps.py               # overlay two DBSCAN-eps processings
│   ├── compare_sig_bkg.py           # matched signal clusters vs background clusters
│   ├── compare_style.py             # shared suep-plot styling for the above
│   └── event_display.py             # r-z rechit event display, coloured by cluster
├── studies/                         # written-up one-off studies (own READMEs)
├── tests/                           # pytest unit tests (no ROOT files needed)
├── reproduce.sh                     # one-shot local reproduction of the MDS plots
├── README_mds.md                    # the MDS LLP cluster study, step by step
└── src/suep_plot/
    ├── _compat.py                   # scipy shim so coffea.lookup_tools imports
    ├── processor.py                 # SuepProcessor(ProcessorABC) + Runner driver
    ├── histograms.py                # build hist.Hist from YAML, per-event & per-object fill
    ├── corrections.py               # correctionlib_wrapper + coffea Weights
    ├── reweights.py                 # expression/binned reweighting + suep-reweight maps
    ├── jme.py                       # jet energy corrections (JEC) + JER smearing
    ├── plot.py                      # mplhep CMS-style plotting (stack, overlay, data)
    ├── slurm.py                     # generates Slurm array job scripts
    ├── status.py                    # completion check + resubmission of failed tasks
    ├── cli.py                       # entry points (run/plot/submit/status/reweight)
    └── cli_worker.py                # single-sample worker invoked by each Slurm task
```

Everything below assumes you run from the repo root
(`/users/ang.li/public/SUEP/suep-plotting`), since config-set and output paths
in the examples are relative to it.

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

`scikit-learn` is additionally needed for the MDS config sets (`custom/columns.py`
runs DBSCAN); it is present in the `mds` env.

> **Note on corrections:** coffea's `lookup_tools` package (which contains
> `correctionlib_wrapper`) eagerly imports Rochester/double-Crystal-Ball modules
> that reference a scipy helper removed in newer scipy. `src/suep_plot/_compat.py`
> restores that helper *before* coffea loads, so no changes to the shared `mds`
> environment are required. See
> [Corrections & the scipy compat shim](#corrections--the-scipy-compat-shim).

---

## Setup

### 1. Install the package

Install once into the env (editable), which puts the console scripts
`suep-run` / `suep-plot` / `suep-submit` / `suep-status` / `suep-reweight` on
your `PATH`:

```bash
conda activate mds
cd /users/ang.li/public/SUEP/suep-plotting
pip install -e .
```

The `-e` (editable) install points at this source tree, so code and config
edits take effect immediately, and `custom/columns.py` stays discoverable.
Uninstall anytime with `pip uninstall suep-plot`.

Without installing, the same commands work as
`PYTHONPATH=src python -m suep_plot.cli {run,plot,submit,status,reweight} …`
from the repo root — this is also what the generated Slurm scripts do, so jobs
need no install.

### 2. Grid proxy (for xrootd inputs)

Most `samples.yaml` entries read `root://eos.grid.vbc.ac.at//...`, which needs a
valid VOMS proxy. Point `X509_USER_PROXY` at it in every shell that runs the
tools (including before `suep-submit`, since the jobs inherit it):

```bash
export X509_USER_PROXY=$HOME/private/.proxy
voms-proxy-info -exists -valid 0:10 || \
  voms-proxy-init -voms cms --valid 192:00 --vomslife 192:0 -out $HOME/private/.proxy
```

Purely local inputs need no proxy.

### 3. Check the install

Unit tests cover fill logic, reweighting and the plot helpers — no ROOT files
needed:

```bash
pip install -e ".[dev]"
pytest tests/
```

A cheap end-to-end smoke test is a one-sample, few-chunk run:

```bash
suep-run -c configs/configs_mds -o /tmp/smoke -s suep_mDark2_temp1 --chunk-size 10000
suep-plot /tmp/smoke -o /tmp/smoke/plots -c configs/configs_mds
```

---

## Quick start

### One command: process + plot

```bash
conda activate mds
cd /users/ang.li/public/SUEP/suep-plotting
export X509_USER_PROXY=$HOME/private/.proxy      # only if inputs are on xrootd
suep-run -c configs/configs_mds --plot
```

This reads every sample in `configs/configs_mds/samples.yaml`, fills every
histogram in `configs/configs_mds/histograms.yaml` (applying selections and
corrections), writes **one pickle file per sample** plus a cutflow table to
`output/`, and renders all figures to `output/plots/`.

`-c` always names one config set — a directory holding the six YAML files.
`configs/` itself is only the container for those sets, so passing it (or
omitting `-c`) is an error that lists what is available.

The real analysis also picks the output directory explicitly — the MDS cluster
study, locally, is:

```bash
suep-run  -c configs/configs_mds -o output_mds --chunk-size 10000 --workers 8
suep-plot output_mds -o output_mds/plots -c configs/configs_mds -j 8
# open output_mds/plots/index.html
```

and [`reproduce.sh`](reproduce.sh) runs that (plus the clustering variants and the
truth-level set) in one go.

Runs are **incremental**: a sample is skipped when its pickle is newer than
the configs, the `custom/` modules, and its input files, so re-running after
adding one sample or histogram only processes what changed. Use `--force`
(`-f`) to reprocess everything (needed after *code* changes, which are not
tracked).

### 1. Process samples and fill histograms

```bash
suep-run -c configs/configs_mds -o output
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
auto by default). Pass the same `-c <config set>` as the run, and **plot-time
styling keys** (`label`, `blind`, `rebin`, `flow`, `log_*`, …) are re-read
from that set's `histograms.yaml` on every invocation — so styling iterations
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
suep-submit -c configs/configs_mds -o output --dry-run   # inspect
suep-submit -c configs/configs_mds -o output             # submit
suep-status -o output                        # what finished, what didn't
bash output/slurm/merge_and_plot.sh          # after jobs finish
```

> **On the login node**, keep local runs small. Many-worker `suep-run` jobs hit
> the per-user process cap and get killed — use `suep-submit` for anything
> beyond a few files per sample.

---

## The commands

Five console scripts; every one takes `-h`. `-c/--config-dir` selects a
[config set](#config-sets-in-this-repo), `-o/--output-dir` the output directory.

| Command | What it does |
|---------|--------------|
| `suep-run` | Read samples, fill histograms, write one pickle per sample (`--plot` to plot straight after). |
| `suep-plot` | Turn pickles into CMS-style figures + cutflow + `index.html` gallery. No reprocessing. |
| `suep-submit` | Write and submit a Slurm array job — one task per sample (or per file shard). |
| `suep-status` | Check a submitted run for missing/truncated pickles; `--resubmit` reruns exactly those tasks. |
| `suep-reweight` | Derive a binned reweight map (ratio of two samples' histograms) from processed pickles. |

### suep-run

```bash
suep-run -c <config dir> -o <output dir> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-c` / `--config-dir` | — (required) | One config set: a directory under `configs/` holding the six YAML files. |
| `-o` / `--output-dir` | `output` | Where per-sample pickles are written. |
| `-s` / `--samples` | all | Process only these sample names. |
| `--chunk-size` | `100000` | Events per chunk (use ~10000 for MDS configs — DBSCAN is memory-hungry). |
| `--workers` | `1` | Local worker processes (coffea `FuturesExecutor`); `1` = iterative. |
| `-f` / `--force` | off | Reprocess even if a pickle looks up to date. |
| `--plot` | off | Run `suep-plot` on the output afterwards, into `<output>/plots`. |
| `--lumi` / `--log` / `--normalize` / `--formats` / `-j` | — | Forwarded to the plotting step (only with `--plot`). |

Runs are **incremental**: a sample is skipped when its pickle is newer than the
configs, the `custom/` modules, and its input files. Python code under
`src/suep_plot/` is *not* tracked — after editing it, pass `--force`.

### suep-plot

```bash
suep-plot <pickles or output dir…> -o <figure dir> [-c <config dir>] [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-o` / `--output-dir` | `plots` | Figure directory (also gets `cutflow.txt/.csv` and `index.html`). |
| `-c` / `--config-dir` | — | Config set to re-read plot-time styling and `derived_plots.yaml` from. |
| `--lumi` | — | Luminosity [fb⁻¹]: CMS label **and** `xs × lumi × 1000 / sumw` MC scaling. |
| `--log` / `--normalize` | off | Log y-axis / normalize signal to unit area. |
| `--no-ratio` | off | Suppress the Data/MC ratio panel. |
| `--formats` | `png,pdf` | Comma-separated output formats (`png` alone is much faster). |
| `-j` / `--jobs` | auto | Parallel rendering processes. |
| `--save-root` | — | Also export merged, scaled histograms to a ROOT file. |

Always pass the `-c` of the config set the pickles came from: labels, colors,
axis labels, `rebin`, `blind`, `log_*` and the derived plots are read at plot
time, so styling iterations never touch the ROOT files.

### suep-submit / suep-status

See [Running on Slurm](#running-on-slurm) for the full flag tables and the
resubmission workflow.

### suep-reweight

```bash
suep-reweight <output dir> --hist <name> --num <sample> --den <sample> -o <map.yaml>
```

Writes a binned map (default: shape-only, clamped) to be referenced from
`reweights.yaml`; see [reweights.yaml](#reweightsyaml).

---

## Config sets in this repo

A *config set* is one directory under `configs/` with the six YAML files.
Samples are pulled from
the shared registry [`datasets.yaml`](datasets.yaml) via `_include`, so all sets
see the same datasets and differ only in what they fill.

| Config set | What it fills | Needs truth? |
|---|---|---|
| [`configs/configs_mds/`](configs/configs_mds/) | Reco-only DBSCAN CSC/DT/RPC cluster properties and shower shapes, ΔR to nearest muon/jet. Fills identically on samples without truth branches. | no |
| [`configs/configs_mds_grid/`](configs/configs_mds_grid/) | Superset of the above plus everything truth-dependent: LLP collection, matched/unmatched cluster splits, efficiency chain, sig-vs-bkg overlays. Runs the full (mDark, T) signal grid. | yes |
| [`configs/configs_mds_gen/`](configs/configs_mds_gen/) | Gen-level only: LLP kinematics, per-LLP matched-rechit counts, ΔR₉₀ maps. Its `columns.yaml` skips DBSCAN. | yes |
| [`configs/configs_mds_data/`](configs/configs_mds_data/) | The same cluster plots on collision data / ZeroBias. | no |
| [`configs/configs_mds_sigonly/`](configs/configs_mds_sigonly/) | Signal samples only — quick turnaround. | yes |
| [`configs/configs_mds_trigger/`](configs/configs_mds_trigger/) | HLT/L1 MDS trigger studies. | yes |
| [`configs/configs_mds_shape/`](configs/configs_mds_shape/) | Cluster shower-shape variables, signal-vs-background overlay in one histogram. | yes |
| [`configs/configs_g4compare/`](configs/configs_g4compare/) | Geant4 / generator comparison of the shower simulation. | yes |

Pick a set by which plots you want, then keep one output directory per
(config set × clustering) combination — the pickles carry no record of which
env knobs produced them. The convention in use:

```
/groups/hephy/cms/ang.li/suep_output/<config>_<gen>_<clustering>/   # suep-run pickles
/groups/hephy/cms/ang.li/suep_plots/<study>_<gen>_<clustering>/     # suep-plot figures
```

e.g. `suep_output/configs_mds_gen3_minpts10_dr04`. Anything matching
`output_*/` in the repo is git-ignored, so local scratch output directories are
fine too.

**Never pass `--lumi` to the MDS config sets:** `xs: 1.0` there is a
placeholder, and lumi scaling would distort the Clopper–Pearson efficiency
intervals.

For the full, step-by-step reproduction of the MDS LLP cluster study (which
samples, which figures, what the plots mean) see
[**README_mds.md**](README_mds.md); [`reproduce.sh`](reproduce.sh) runs the local
version of it end to end.

---

## Derived-column settings (`columns.yaml`)

The knobs of `custom/columns.py` belong in the config directory, next to
`histograms.yaml`, as an optional `columns.yaml`:

```yaml
parameters:
  cluster_eps: 0.4            # DBSCAN eps — the ΔR radius in η–φ, all systems
  cluster_min_samples: 10     # DBSCAN min_samples for CSC/DT (50 = standard MDS)
  rpc_min_samples: 10         # DBSCAN min_samples for RPC (sparse system)
  match_min_hits: 10          # cluster ↔ LLP truth-match threshold
  dr_quantiles: [0.5, 0.8, 0.9]   # → llp.dr50/dr80/dr90
  pair_max_hits: 2000
  llpidx_convention: genpart  # or 'ordinal' (pre-Geant4-fix files)
  # Which prompt objects the cluster ΔR is measured against
  jerc: true                  # JEC (L1L2L3Res) on events.Jet + JER smearing on MC
  jerc_era: 2024_Summer24     # jsonpog payload directory
  jerc_algo: AK4PFPuppi
  jerc_data_tag:              # null = the era's jec_tag_data from suep_plot.jme
  iso_objects:                # one dR field per entry, selection written here
    drMuon:
      collection: Muon
      expression: "(obj.pt > 10) & (abs(obj.eta) < 2.4) & obj.looseId"
    drJet:
      collection: Jet
      expression: "(obj.pt > 20) & (abs(obj.eta) < 2.4) & (obj.neHEF < 0.8) & (obj.chHEF > 0.1) & jet_id(obj, 'tightlepveto')"
steps: [jerc, clusters, cluster_isolation, llp, llp_hits, llp_reco, llp_shape]
```

Both blocks are optional; anything left out keeps the default shown above.
The file is part of the run's identity — `suep-run` treats it like the other
configs for the up-to-date check, so editing it reprocesses.

`derive()` reads these settings once, at entry, and passes them down to the
helpers, so nothing below it depends on module state. Nothing is installed at
import time either: `configure()` has to run in **every process** that calls
`derive()` — the CLI does it in the parent and again in each worker — and a
process that skipped it raises on the first parameter read rather than falling
back on the defaults. That distinction is the whole point: a worker silently
running the defaults while the config asked for something else is a run whose
histograms are wrong and whose logs look fine. A `columns.yaml` that fails
validation installs nothing, so the settings already in force stay in force.
An interactive user (a notebook, `scripts/event_display.py`) calls
`custom.columns.configure()` with no argument for the defaults.

`steps` selects which of the optional helpers `derive()` runs (the default is
all of them):

| Step | Attaches | Needs |
|------|----------|-------|
| `jerc` | nothing — rescales `events.Jet` pT/mass in place (JEC on data and MC, JER smearing on MC) | — |
| `clusters` | `events.<sys>Cluster` (DBSCAN, ~35 % of `derive()`) | — |
| `cluster_isolation` | `drMuon` / `drJet` on the clusters, measured against the selected muons/jets | `clusters` |
| `llp` | `events.llp`: kinematics, decay vertex, volume flags | — |
| `llp_hits` | `llp.nHits{CSC,DT,RPC,Total}` | `llp` |
| `llp_reco` | `llp.reco*`, `nRecoCluster*`, `clusterHitFrac*` | `clusters`, `llp_hits` |
| `llp_shape` | `llp.dr*` / `drPair*`, rechit `matchedLLP` / `drLLP` | `llp` |

A step you leave out means its fields are *absent*, so a config referencing
them fails at expression validation instead of quietly filling empty
histograms. Unknown keys, unknown steps and unmet dependencies are fatal.
See [`configs/configs_mds_gen/columns.yaml`](configs/configs_mds_gen/columns.yaml) for the
gen-level-only set.

#### Which muons and jets count as prompt activity

`cluster_isolation` measures `drMuon` / `drJet` to the closest **selected**
object, so a cluster sitting next to a muon that fails the ID counts as
isolated rather than vetoed. Events with no selected muon (jet) at all get the
sentinel `NO_OBJECT_DR = 999`, which passes any isolation cut and lands in the
overflow of the ΔR histograms.

Two details are specific to 2024 NanoAOD:

- **The object selections live in the config, not in the code.** `iso_objects`
  maps each dR field to a `collection` plus a per-object boolean `expression`,
  evaluated by a single `_selected_objects(events, selection)`. Adding a field —
  electrons, photons, HLT jets — is a config edit; nothing in
  `custom/columns.py` knows about muons or jets specifically. In scope: `obj`
  (the collection), `events`/`ev`, `ak`, `np`, the safe builtins, and
  `jet_id(obj, "tight"|"tightlepveto")`.
- **`Jet_jetId` is not stored any more**, which is why `jet_id()` exists: it
  evaluates the ID from the PF energy fractions and multiplicities with the
  official jsonpog `jetid.json.gz` (`AK4PUPPI_Tight` /
  `AK4PUPPI_TightLeptonVeto`), so the thresholds stay in the central payload.
- **JEC/JER rescale pT and mass only** — η and φ are untouched — so `jerc`
  reaches ΔR *only* through the `iso_jet_pt` threshold. (`correct_jets` re-sorts
  jets by the new pT, so the ordering moves, but ΔR-to-nearest is
  order-invariant.)
- **Data is corrected too**, with the era's `*_DATA` tag — `L1L2L3Res` is where
  the residual corrections that exist for data live. Only the JER smearing is
  MC-only, and `correct_jets` already switches it off when `GenJet` is absent.
  The 2024 DATA compound declares an extra `run` input: one payload covers the
  whole year and picks the residual IOV per event, so no run-range map is
  needed. MC vs data is decided by `genWeight` (`has_truth` is False for the
  non-SUEP MC too, so it cannot be used for this).

### Environment knobs

| Variable | Default | Effect |
|----------|---------|--------|
| `X509_USER_PROXY` | — | Grid proxy for xrootd reads (see [Setup](#2-grid-proxy-for-xrootd-inputs)). |
| `CORRECTIONLIB_DATA` | — | Searched first for `auto:` correction payloads, before cvmfs jsonpog-integration. |
| `MPLBACKEND` | set to `Agg` | Forced headless by the CLI unless you override it. |

Nothing about `derive()` is settable from the environment: the derived-column
settings live in the config set, so a run is reproducible from its config
directory alone and nothing has to be exported before `suep-run` or
`suep-submit`. (The `MDS_CLUSTER_MIN_SAMPLES` / `MDS_CLUSTER_EPS` /
`MDS_SKIP_CLUSTERING` / `MDS_LLPIDX_CONVENTION` variables older runs used are
gone — put the values in `columns.yaml`. Output directories filled before the
switch used the defaults above unless their command line says otherwise.)
Because a rerun re-reads the config, `suep-status --resubmit` warns if
`columns.yaml` was edited after the run was submitted.

```bash
# a variation on an existing config set: copy it, edit columns.yaml
cp -r configs/configs_mds configs/configs_mds_minpts50
sed -i 's/cluster_min_samples: 10/cluster_min_samples: 50/' configs/configs_mds_minpts50/columns.yaml
suep-run -c configs/configs_mds_minpts50 -o output_mds_minpts50 --chunk-size 10000
```

---

## Configuration reference

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

Datasets are defined once in the top-level [`datasets.yaml`](datasets.yaml); each
config picks the combination it needs instead of repeating the definitions:

```yaml
_include: ../../datasets.yaml    # one path or a list; relative to this samples.yaml

suep_mDark2_temp1:            # registry entry, verbatim
suep_temp2:                   # registry entry with per-study overrides
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

## Running locally

```bash
# all samples (skips ones whose pickle is already up to date)
suep-run -c configs/configs_mds -o output --chunk-size 200000

# one sample
suep-run -c configs/configs_mds -o output -s suep_mMed125_mDark2

# multi-core (coffea FuturesExecutor)
suep-run -c configs/configs_mds -o output --workers 4

# reprocess everything (e.g. after editing suep_plot code)
suep-run -c configs/configs_mds -o output --force

# process + plot in one go
suep-run --plot --lumi 38.5 --log

# MDS config sets: small chunks (DBSCAN runs per chunk); clustering knobs
# come from configs/configs_mds/columns.yaml
suep-run -c configs/configs_mds -o output_mds --chunk-size 10000 --workers 8
```

Local runs are for one sample, a few files, or a styling iteration. Anything
larger belongs on Slurm: the login node's per-user process cap kills
many-worker `suep-run` jobs, and a full sample over xrootd takes hours.

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

The full cycle:

```bash
conda activate mds
export X509_USER_PROXY=$HOME/private/.proxy      # jobs inherit this
# clustering knobs need no exporting: they live in configs/configs_mds/columns.yaml

suep-submit -c configs/configs_mds -o output_mds --conda-env mds \
    --time 08:00:00 --mem 8000 --partition c \
    --chunk-size 10000 --files-per-job 5 --max-concurrent 50 --dry-run   # inspect
suep-submit -c configs/configs_mds -o output_mds --conda-env mds \
    --time 08:00:00 --mem 8000 --partition c \
    --chunk-size 10000 --files-per-job 5 --max-concurrent 50             # submit

squeue -u $USER                                  # watch
suep-status -o output_mds                         # missing / truncated pickles
suep-status -o output_mds --resubmit              # rerun exactly those tasks
bash output_mds/slurm/merge_and_plot.sh            # once complete -> output_mds/plots
```

| Flag | Default | Description |
|------|---------|-------------|
| `-s` / `--samples` | all | Submit only these samples (e.g. signal-only reruns). |
| `--conda-env` | `mds` | Environment activated inside each job. |
| `--time` / `--mem` | `04:00:00` / `8000` | Wall time / memory (MB) per job. |
| `--partition` / `--account` | — | Slurm partition / account. |
| `--chunk-size` | `100000` | Events per chunk. |
| `--workers` | `1` | Worker processes per job (also sets `--cpus-per-task`). |
| `--files-per-job` | — | Split each sample into tasks of this many files; parts (`<sample>.part<k>.pkl`) are summed by `suep-plot`. |
| `--max-concurrent` | — | Cap on simultaneous array tasks (`%N`). |
| `--dry-run` | off | Generate scripts without submitting. |

`--files-per-job` is how you put more nodes in flight on a big sample: 25 files
with `--files-per-job 5` becomes 5 tasks instead of 1. The parts merge
automatically at plot time, so nothing downstream changes.

Sample files are resolved **once, at submission time**: each task's files are
written to `slurm/filelists/<sample>[.part<k>].txt` and the task reads that list
instead of expanding `files:` itself. So a directory that gains files after
submission cannot shift shard boundaries, tasks don't re-list storage, and a
resubmission reruns exactly the same inputs. To pick up new files, submit again.

Slurm jobs always reprocess their sample (`--force`); the incremental skip only
applies to local `suep-run`.

### Checking a run, and redoing what failed

A task counts as done only when its pickle exists **and unpickles** — a task
killed mid-write leaves a truncated file that would otherwise be summed into the
merge as a silently incomplete sample. `suep-status` checks both against
`slurm/task_list.txt` (the authority on what should exist):

```bash
suep-status -o output_mds                    # report only
suep-status -o output_mds --no-verify        # existence check only (fast, large runs)
suep-status -o output_mds --resubmit --dry-run
suep-status -o output_mds --resubmit --max-concurrent 50
```

`--resubmit` runs `sbatch --array=<missing indices>` over the *original*
`job.sh`, so the redone tasks own exactly the same input files and shard
boundaries as the first attempt. Export the same
[environment knobs](#environment-knobs) you submitted with — `job.sh` inherits
the calling shell, and `suep-status` warns when they are not pinned in the
script.

Resubmitting by hand also works; the task index is the line number − 1 in
`slurm/task_list.txt`:

```bash
sbatch --array=3,7 output_mds/slurm/job.sh
```

### Adding up the per-task outputs

There is no separate merge step to run: `suep-plot` sums whatever pickles it is
given. Each `hist.Hist` carries a `dataset` string axis, so adding pickles keeps
samples apart, while several pickles of the *same* sample accumulate into the
same bin:

```bash
bash output_mds/slurm/merge_and_plot.sh        # = suep-plot output_mds -o output_mds/plots -c configs/configs_mds
suep-plot output_mds -o output_mds/plots -c configs/configs_mds -j 8      # the same, by hand
```

`merge_results()` globs `*.pkl` from the directory and sums the histograms
together with each sample's `sumw`, `nevents` and cutflow counts. So a
`--files-per-job` run's `suep_mDark2_temp1.part0.pkl … part4.pkl` become one
`suep_mDark2_temp1` with correct `--lumi` normalization and a correct cutflow — no
renaming or concatenating needed. Pickles from *different* runs merge just as
well (`suep-plot output_a output_b -o cmp`), as long as the histogram axes agree.

**Check completeness first.** A missing or truncated part is not an error at
merge time, it just undercounts that sample — which is why `suep-status`
verifies each pickle unpickles:

```bash
suep-status -o output_mds && bash output_mds/slurm/merge_and_plot.sh
```

To get one merged object outside the plotter, use `--save-root merged.root`
(`<histogram>/<sample>` TH1Ds) or call the same helper:

```python
from suep_plot.plot import merge_results, _resolve_inputs
data = merge_results(_resolve_inputs("output_mds"))
h = data["histograms"]["csc_cluster_size"][{"dataset": "suep_mDark2_temp1"}]
print(data["sumw"], data["nevents"])
```

### What `suep-submit` writes

```
<output>/slurm/
├── job.sh                  # the array script (exports PYTHONPATH=<repo>/src)
├── task_list.txt           # one line per array index: sample, file list, part
├── filelists/<sample>[.part<k>].txt
├── logs/                   # stdout/stderr per task
└── merge_and_plot.sh       # suep-plot over the finished pickles
```

`merge_and_plot.sh` already carries the run's `-c <config dir>` and
`-o <output>/plots`, and forwards any extra arguments to `suep-plot`
(`bash …/merge_and_plot.sh --log --formats png`).

---

## Helper scripts

Standalone tools under `scripts/`. They read the same pickles `suep-plot` reads
and reuse its styling helpers (`compare_style.py`), so figures come out with the
same CMS style, labels and png+pdf gallery.

**Compare two clustering settings** — one figure per 1D histogram per sample:
unit-normalized shapes for both settings, ratio (b/a) underneath, yields in the
legend:

```bash
python scripts/compare_eps.py output_mds_dr02 output_mds_dr04 compare_dr02_vs_dr04 "dR=0.2" "dR=0.4"
```

**Signal (truth-matched) clusters vs background clusters** — the signal's
`<var>_matched` histograms come from a `configs/configs_mds_grid` run, the background's
inclusive `<var>` from a `configs/configs_mds` run; both configs define the inclusive
histograms identically, so the axes match and nothing needs refilling. Also
writes `separation.txt` (total-variation distance per variable):

```bash
python scripts/compare_sig_bkg.py output_mds_grid output_mds_data sigbkg_dir \
    --suffix _matched -c configs/configs_mds_grid
```

**Event display** — r–z picture of every CSC/DT rechit, coloured by DBSCAN
cluster, using the analysis's own clustering (`-c <config set>` reads that
set's `columns.yaml`, so the display clusters exactly as its histograms were
filled):

```bash
conda activate mds
export X509_USER_PROXY=$HOME/private/.proxy
python scripts/event_display.py -c configs/configs_mds -d suep_mDark2_temp1 -n 5 --matched-only --min-size 50 --zoom --with-etaphi
```

Useful flags: `--zoom` (crop to the clustered hits), `--with-etaphi` (add the
η–φ panel the clustering runs in), `--matched-only` / `--min-size` (pick
interesting events), `--systems csc`, and `-f <file.root> --entries 3 7` for
specific entries. It works on background/data too — those have no truth
branches, so every cluster is labelled "unmatched". See
[README_mds.md](README_mds.md#event-display-rz-rechit-picture-of-the-clusters)
for what the markers mean.

`studies/` holds written-up one-off analyses (each with its own README and
scripts) that are not part of the config-driven flow.

---

## How it works internally

```
configs/<set>/*.yaml
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
corrected jets. The **JEC** is coffea's `CorrectedJetsFactory`, fed the jsonpog
`jet_jerc.json.gz` payloads through coffea's `correctionlib_adapters`. The
**JER smearing** is the official `JERSmear` payload (`jer_smear.json.gz`, shipped
with the package), evaluated per jet with the gen pT of jets matched within
`dR < 0.2` and `3σ` — that is what switches it between gen-matched scaling and
its deterministic, event-seeded stochastic mode; a non-finite or non-positive
factor falls back to 1.0. `suep_plot/jme.py` supplies the wiring: the payload
tags (including the run-dependent `*_DATA` compound), the per-jet
`pt_raw`/`event_rho`/`run` columns, the Type-1 MET rebuild and the pT re-sort.

**MET is rebuilt, not patched:** the Type-1 correction is recomputed from
`RawPuppiMET` as the vector sum of `pT_L2L3Res − pT_L1` over the muon-subtracted
jets of *both* collections — `Jet` and `CorrT1METJet`, the sub-15 GeV jets
NanoAOD stores precisely for this — projected along `φ + muonSubtrDeltaPhi`,
the axis of the jet once its muon is removed (bare `φ` when a dataset lacks that
branch), with the standard `pT_corr > 15`,
`|η| < 5.2`, `EM fraction < 0.9` selection, and with whatever the jets picked up
beyond the nominal JEC (smearing, a JES variation) carried into the sum. Feeding
the *production* jet pT through the same code reproduces the production Type-1
term to 0.001 GeV — that closure is what validates it. Rebuilding with the
current calibration lands ~6 GeV from the stored `PuppiMET`, because the JEC has
moved since sample production (stored jets carry a mean factor of 1.58 versus
1.25 for `Summer24Prompt24_V5`); that is the intended difference, not an error.

Payloads come from the **CAT campaign directory** for the era —
`/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/<campaign>/latest` — which
`suep_plot.jme.payload_path()` prefers over cvmfs jsonpog-integration
(`$CORRECTIONLIB_DATA` still wins over both). This is not cosmetic: jsonpog's
`2024_Summer24` only carries `Summer24Prompt24_V1`, while the 2024 recommendation
is `V5` (`V3`→`V4` bumped the tag without updating the L2L3Residual payloads;
`V5` is the fix). `custom/columns.py` resolves the jet ID through the same
function, so the ID and the calibration cannot drift onto different campaigns.
Systematics are one argument away:

```python
events = correct_jets(events, variation="jec_up")    # jec_down / jer_up / jer_down
```

The jet-momentum change is also **Type-1 propagated to the MET**: the vector
sum of `p_new − p_stored` over jets with corrected pT > 15 GeV and EM
fraction < 0.9 is subtracted from `PuppiMET` (`met="PFMET"` to target another
collection, `met=None` to disable), so `events.PuppiMET.pt` selections and
histograms respond consistently to JEC/JER variations.

For data pass the run-specific tag and disable smearing:
`correct_jets(events, jec_tag="Summer24Prompt24_RunX_V1_DATA", smear=False)`.
Payload files resolve from `$CORRECTIONLIB_DATA`, then cvmfs
jsonpog-integration. `pt`/`mass`/MET are replaced in place (re-run with the
example removed to get uncorrected values — `custom/` edits are
tracked, so affected samples re-run automatically).

**Custom derived columns** — edit `custom/columns.py`; `derive(events)` returns the
(augmented) events array. Attach fields with `ak.with_field(events, value, "name")`
and reference them as `events.name` in any expression. Its helpers are split by
topic over `custom/params.py`, `clustering.py`, `llp.py` and `isolation.py`, all
re-exported from `custom.columns`.

**xrootd files** — list `root://host//store/…` URLs under `files:`; coffea/uproot
handle them natively (ensure a valid grid proxy / kerberos token).  An entry may
also be a *directory* URL (`root://host//store/user/…/MDSNANO`), which is listed
on the server and walked recursively for `*.root`, or a wildcard on the file
name (`…/MDSNANO/nano_*.root`) — so a whole dataset is one line instead of
hundreds.  Local directories work the same way.  Listing happens each time
`suep-run`/`suep-slurm` starts, so files added later are picked up automatically.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `suep-run` reports "up to date" and does nothing | Incremental skip: only configs, the `custom/` modules and input files are tracked, not `src/suep_plot/` code. Pass `--force`. |
| xrootd errors / "no such file" on `root://eos.grid.vbc.ac.at` | Expired or unset proxy. `export X509_USER_PROXY=$HOME/private/.proxy` and re-run `voms-proxy-init` (see [Setup](#2-grid-proxy-for-xrootd-inputs)). Don't force `XrdSecPROTOCOL`. |
| Local run dies with process/fork errors | Login-node process cap — drop `--workers`, or submit with `suep-submit`. |
| Killed for memory | Lower `--chunk-size` (10000 is right for the MDS configs, where DBSCAN runs per chunk), or raise `--mem` on Slurm. |
| Warning listing expressions that failed the check | Typo or missing branch. The processor validates every expression once on a small slice with `derive()` applied — fix the expression, or the histogram fills empty. |
| `ERROR: configs is not a config set` | `-c` must name one set, e.g. `configs/configs_mds`; `configs/` only holds them. The message lists the available sets. |
| `configs/configs_mds_gen` fails at validation on `events.<sys>Cluster` | Expected: its `columns.yaml` omits the `clusters` step, so the cluster collections are deliberately absent. Use a config set that doesn't reference them, or add the step. |
| `ERROR in columns.yaml: ...` | Unknown key/parameter/step, or a step whose dependency is not enabled — see the table in [Derived-column settings](#derived-column-settings-columnsyaml). Fatal by design: a typo here would mean silently different histograms. |
| Merged output looks inconsistent | Shards filled with different derived-column settings. Keep the knobs in the config set's `columns.yaml` (not in `$MDS_*`) and one output directory per (config set × clustering). |
| Plots have no labels / wrong colors | `suep-plot` was called without `-c <config set>`, so styling fell back to defaults. |
| Empty or truncated pickles after a Slurm run | `suep-status -o <dir>` finds them; `--resubmit` reruns exactly those tasks. |
| Efficiency plots look distorted | `--lumi` was passed to an MDS config set (placeholder `xs: 1.0`). Drop it. |
| `ImportError: _lazywhere` from `coffea.lookup_tools` | `suep_plot` was bypassed — import `suep_plot` (or the shim) first; see [the compat shim](#corrections--the-scipy-compat-shim). |
