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
3. [Setup](#setup)
4. [Quick start](#quick-start)
5. [The commands](#the-commands) — `suep-run`, `suep-plot`, `suep-submit`, `suep-status`, `suep-reweight`
6. [Config sets in this repo](#config-sets-in-this-repo)
7. [Running locally](#running-locally)
8. [Running on Slurm](#running-on-slurm)
9. [The MDS LLP cluster study](#the-mds-llp-cluster-study) — the analysis this package was written for
10. [Helper scripts](#helper-scripts)
11. [How it works internally](#how-it-works-internally)
12. [Corrections & the scipy compat shim](#corrections--the-scipy-compat-shim)
13. [Environment knobs](#environment-knobs)
14. [Recipes](#recipes)
15. [Troubleshooting](#troubleshooting)

**Reference docs:**
[`docs/configuration.md`](docs/configuration.md) — every YAML file and directive ·
[`docs/derived-columns.md`](docs/derived-columns.md) — `custom/`: the LLP and cluster collections and their settings

---

## Repository layout

```
suep-plotting/
├── pyproject.toml                   # package metadata, dependencies, console scripts
├── datasets.yaml                    # central dataset registry (paths, xs, labels)
├── configs/                         # one subdirectory per config set
│   ├── _common/                     # fragments several sets _include
│   │   └── prompt_objects.yaml      #   the muon/jet cuts the isolation uses
│   ├── configs_mds/                 # a "config set" = these files
│   │   ├── samples.yaml             # which datasets to run, cross sections, styling
│   │   ├── histograms.yaml          # histogram definitions (NanoEvents expressions)
│   │   ├── selections.yaml          # named event-/object-level cuts
│   │   ├── corrections.yaml         # correctionlib scale-factor definitions
│   │   ├── reweights.yaml           # event-/object-level reweighting (expressions & maps)
│   │   ├── derived_plots.yaml       # profiles/projections/efficiency/ratio at plot time
│   │   └── columns.yaml             # optional: parameters + enabled steps of derive()
│   └── configs_mds_signal/  configs_mds_gen/  configs_mds_data/  …
│                                    # the other config sets (see "Config sets")
├── custom/                          # derive(events) -> events hook: LLP + DBSCAN
│   ├── columns.py                   #   derive() itself + the step pipeline
│   ├── params.py                    #   PARAM_SPEC: the columns.yaml schema
│   ├── clustering.py                #   DBSCAN of one rechit system
│   ├── llp.py                       #   events.llp + per-LLP rechit spread
│   └── helpers.py                   #   object selection, isolation dR, jet ID
├── scripts/                         # standalone plotting/inspection tools
│   ├── compare_eps.py               # overlay two DBSCAN-eps processings
│   ├── compare_sig_bkg.py           # matched signal clusters vs background clusters
│   ├── compare_style.py             # shared suep-plot styling for the above
│   ├── event_display.py             # r-z rechit event display, coloured by cluster
│   ├── dump_config.py               # print a config set fully expanded (refactor check)
│   └── smoke_test.sh                # ~1 min end-to-end run+plot over a single file
├── docs/                            # configuration.md (YAML reference),
│                                    # derived-columns.md (custom/ reference)
├── studies/                         # written-up one-off studies (own READMEs)
├── tests/                           # pytest unit tests (no ROOT files needed)
├── reproduce.sh                     # one-shot local reproduction of the MDS plots
└── src/suep_plot/
    ├── config.py                    # the one YAML reader: _extends / _repeat / _include / _registry
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
`reweights.yaml`; see [reweights.yaml](docs/configuration.md#reweightsyaml).

---

## Config sets in this repo

A *config set* is one directory under `configs/` with the six YAML files (plus
an optional `columns.yaml`). Samples are pulled from the shared registry
[`datasets.yaml`](datasets.yaml) via `_registry:`, so all sets see the same
datasets and differ only in what they fill. Sets that are variations on another
say so with `_extends:` and carry only the difference — see
[`docs/configuration.md`](docs/configuration.md).

| Config set | What it fills | Needs truth? |
|---|---|---|
| [`configs/configs_mds/`](configs/configs_mds/) | Reco-only DBSCAN CSC/DT/RPC cluster properties and shower shapes, ΔR to nearest muon/jet. Fills identically on samples without truth branches. | no |
| [`configs/configs_mds_signal/`](configs/configs_mds_signal/) | Superset of the above plus everything truth-dependent: LLP collection, matched/unmatched cluster splits, efficiency chain, sig-vs-bkg overlays. Runs the full (mDark, T) signal grid. | yes |
| [`configs/configs_mds_gen/`](configs/configs_mds_gen/) | Gen-level only: LLP kinematics, per-LLP matched-rechit counts, ΔR₉₀ maps. Its `columns.yaml` skips DBSCAN. | yes |
| [`configs/configs_mds_data/`](configs/configs_mds_data/) | The same cluster plots on collision data / ZeroBias. | no |
| [`configs/configs_mds_sigonly/`](configs/configs_mds_sigonly/) | Signal samples only — quick turnaround. | yes |
| [`configs/configs_mds_trigger/`](configs/configs_mds_trigger/) | HLT/L1 MDS trigger studies. | yes |
| [`configs/configs_mds_shape/`](configs/configs_mds_shape/) | Cluster shower-shape variables, signal-vs-background overlay in one histogram. | yes |
| [`configs/configs_g4compare/`](configs/configs_g4compare/) | Geant4 / generator comparison of the shower simulation. | yes |
| [`configs/configs_mds_rpcmerge/`](configs/configs_mds_rpcmerge/), [`_rpcmerge_data/`](configs/configs_mds_rpcmerge_data/) | `configs_mds_signal` / `configs_mds_data` with `rpc_mode: merge` — RPC rechits clustered with the system they overlap, plus the RPC timing histograms. `_extends` the reference set, so the two stay comparable plot for plot. | as the base |
| [`configs/configs_mds_rpcmatch/`](configs/configs_mds_rpcmatch/), [`_rpcmatch_data/`](configs/configs_mds_rpcmatch_data/) | The same with `rpc_mode: match` — the clustering is untouched and RPC only dates the finished clusters. | as the base |
| [`configs/configs_smoke/`](configs/configs_smoke/) | Not a study: one signal file, one histogram, `steps: [clusters]`. What [`scripts/smoke_test.sh`](scripts/smoke_test.sh) runs to prove the chain works after a framework change, in ~1 min. | no |

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
[**the MDS study section below**](#the-mds-llp-cluster-study);
[`reproduce.sh`](reproduce.sh) runs the local version of it end to end.

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

## The MDS LLP cluster study

Analysis of two SUEP MDSNANO signal points (ggH, mMed=125, mDark=2, cτ=5 m,
`temp=1` / `temp=2`) read over xrootd from VBC EOS. The derived collections
(LLPs, DBSCAN CSC/DT/RPC clusters, cluster↔LLP matching) are built in
[`custom/columns.py`](custom/columns.py); the plots are split over two config
sets by whether they need gen information:

| config | contents | output |
| --- | --- | --- |
| [`configs/configs_mds/`](configs/configs_mds/) | reco-only DBSCAN cluster properties and shower shapes, incl. ΔR to the closest muon / jet — fills identically on samples without truth branches | `output_mds*/` |
| [`configs/configs_mds_signal/`](configs/configs_mds_signal/) | everything truth-dependent: the LLP collection, the truth-matched/unmatched cluster splits, the efficiency chain, the matched fractions and the signal-vs-background overlays; runs the full (mDark, T) grid | `output_mds_grid/` |

Both use the same `derive()`, so pick the config set
by which plots you want; `configs/configs_mds_signal/` is the superset of the fills.

Two configurations are produced:

| folder | DBSCAN min cluster size (CSC/DT) | how |
| --- | --- | --- |
| `output_mds/` | 10 | `configs/configs_mds/columns.yaml` |
| `output_mds_minpts50/` | 50 (standard MDS) | copy of that config with `cluster_min_samples: 50` |

RPC clustering is always `min_samples=10` (sparse system). **Never pass `--lumi`** —
`xs=1.0` in `configs/configs_mds/samples.yaml` is a placeholder and lumi scaling would
distort the Clopper–Pearson efficiency intervals.

---

### RPC with the CSC and DT clusters (a time for each shower)

RPC is the only muon subdetector whose MDSNano rechits carry timing at all, so
it is what can date a CSC or DT cluster.  The barrel wheels (`Region == 0`) sit
at the DT radii and the endcap disks (`|Region| == 1`) at the CSC z, so each RPC
rechit pairs with one of the two.  There are two ways to use that, chosen by one
line in `columns.yaml`, and both drop `events.rpcCluster` — every RPC rechit
already belongs to a CSC or a DT cluster, so clustering it again on its own
would double count it:

| `rpc_mode` | config sets | what happens |
| --- | --- | --- |
| `merge` | `configs_mds_rpcmerge/`, `configs_mds_rpcmerge_data/` | the RPC rechits go **into the DBSCAN** with the system they overlap, so a shower crossing both detectors is one cluster instead of two |
| `match` | `configs_mds_rpcmatch/`, `configs_mds_rpcmatch_data/` | the DBSCAN is exactly the reference CSC/DT one and the RPC rechits are **associated afterwards** — nearest cluster centroid within `cluster_eps` — so they date a cluster without being able to create one |

```yaml
parameters:
  rpc_mode: merge      # or: match
```

Both attach the same fields to the same two collections, so one set of plots
reads either and the two can be compared plot for plot.  The difference that
matters is what RPC is allowed to do to the *clustering*: under `merge` an RPC
rechit counts towards `min_samples`, and in data only ~19 % of RPC rechits sit
at BX 0 (essentially flat over −2…+2), so RPC noise can push a background
cluster over threshold.  Under `match` it cannot, and every cluster variable is
bit-identical to a reference run.  Merging in exchange joins a shower that
straddles the two detectors, and lets RPC hits bridge a gap in the CSC/DT hit
density.

Positional resolution is *not* what separates them: measuring the ΔR spread of
gen-matched rechits about the shower centroid on one signal file, endcap RPC is
1.4× wider than CSC (0.053 vs 0.037) and barrel RPC is slightly *tighter* than
DT (0.046 vs 0.057) — both far inside `eps = 0.4`.

Each cluster gains its RPC content (`nRPCHits`, `rpcHitFrac`,
`firstSystem`) and the RPC estimate of **when** it happened:

| field | meaning |
| --- | --- |
| `rpcTime`, `rpcTimeMedian`, `rpcTimeSpread` | mean / median / RMS of the RPC rechit times, over the hits with a valid one |
| `rpcTimeWeighted`, `rpcTimeErr` | `1/σ²`-weighted mean from the per-hit `TimeError`, and its uncertainty |
| `rpcTimeValidFrac` | fraction of the cluster's RPC hits that had a valid time at all |
| `rpcBx`, `rpcBxMedian`, `rpcBxSpread` | mean / median / RMS bunch crossing (× 25 ns for a time) |
| `rpcOutOfTimeFrac`, `nRPCHitsBx0` | fraction of the cluster's RPC hits outside the in-time BX, and the count inside it |

All of them are NaN on a cluster with no RPC hit (the `<sys>_cluster_has_rpc`
selection filters those out), and the CSC `time` (mean `Tpeak`) is unchanged by
either mode: the two detectors are on different clocks, so the RPC estimate is
reported apart rather than averaged in.  `rpcHitFrac` means the same thing in
both — the RPC share of the cluster's hits — which under `match` is
`n / (size + n)`, since matching leaves those hits out of `size`.
`firstSystem` is always the primary system under `match`, so the
`first_not_rpc` selections simply pass everything there.

**The rest of this section describes `rpc_mode: merge` specifically.**  The
merged-in RPC hits keep their
own layer and chamber ids (offset by +1000 in `firstChamber`, so a cluster whose
innermost hit is an RPC one lands in the overflow of the chamber-code plots and
`firstSystem` says so); the CSC/DT hits keep the ids they always had, so
`nLayer`, `firstChamber` and `firstStation` stay comparable with the reference
run except for the RPC hits now inside the cluster.  The `first_chamber` /
`first_station` histograms are cut to clusters whose innermost hit belongs to
the primary system (`<sys>_cluster_first_not_rpc`) — on this file that is 93 %
of the CSC clusters but only 70 % of the DT ones, since the barrel RPC layers
sit inside the DT stations; `<sys>_cluster_first_system` counts all three.

> **The rechit time is not filled in any MDSNano production so far.**
> `rpcRecHits_Time` is 0 with `TimeError` −1 throughout — checked on the Gen3
> signal, on DY and on ZeroBias 2024C — so every `rpcTime*` field is NaN and its
> histograms come out empty by construction (that is deliberate: averaging the
> placeholder zeros would report every cluster as perfectly in time).  The
> **bunch crossing is filled** and carries real structure, so today's RPC time
> estimate is `rpcBx * 25` ns, plotted as `<sys>_cluster_rpc_time_from_bx`.
> Filling `rpcRecHits_Time` in the ntuplizer is what would turn the fine-time
> fields on; nothing here has to change for that.

### Rejecting out-of-time background

Timing here rejects *out-of-time* activity — previous/later-BX pile-up, cavern
background, noise. In-time pile-up is not a timing problem; that is what the
isolation and shape variables are for.

On **CSC** the handle is the rechit time, and it is the cluster's *spread*, not
its mean, that discriminates (on one signal file: gen-matched clusters have an
RMS of 5.9 ns, unmatched ones 18.3 ns, with both means at ~1.5 ns). Every
cluster of a system whose rechits carry a time therefore gets

| field | meaning |
| --- | --- |
| `time` | mean rechit time (CSC `Tpeak`) — unchanged |
| `timeSpread` | RMS of those times over the cluster |
| `ootHitFrac` | fraction of them outside ±`oot_time_cut` (12.5 ns by default) |

On **DT** there is no rechit time at all, so the equivalent handle is the RPC
content of the merged cluster — `rpcBxSpread`, `rpcOutOfTimeFrac`,
`nRPCHitsBx0` — which is how the standard CMS DT muon-shower search times its
clusters. The same split shows up there: matched clusters have a BX RMS of
0.31 against 0.82 for unmatched ones, while both means sit at ~0.1.

> **Do not take the rejection factor from MC.** The BX composition is wildly
> different in data: BX = 0 holds 77 % of the RPC hits in the signal MC, 61 %
> in DY, but only 19 % in ZeroBias 2024C, where the distribution is essentially
> flat over −2…+2 (an unmodelled uniform cavern/noise background filling the
> readout window). The CSC picture matches: `|Tpeak| < 12.5` ns keeps 61 % of
> signal-MC rechits and 22 % of ZeroBias ones. The real rejection is therefore
> much *larger* than MC suggests — measure it in ZeroBias, and take only the
> signal efficiency from MC.

The two config sets are **generated**, not hand-written — they are
`configs_mds_signal/` and `configs_mds_data/` with the RPC-cluster blocks pruned,
the "reconstructable" hit counts widened to the merged system
(`nHitsCSC + nHitsRPCEndcap`, `nHitsDT + nHitsRPCBarrel`) and the RPC timing
histograms added, so everything the two studies share stays identical by
construction:

```bash
```

Running them is the usual pair of commands:

```bash
suep-run  -c configs/configs_mds_rpcmerge -o output_mds_rpcmerge --chunk-size 10000 --workers 8
suep-plot output_mds_rpcmerge -o output_mds_rpcmerge/plots -c configs/configs_mds_rpcmerge -j 8
```

---

### What lives where

- [`configs/configs_mds/`](configs/configs_mds/) — the reco-only fills: cluster
  properties and shower shapes, incl. `<sys>_cluster_dr_muon` / `_dr_jet`, and
  the reco-only event masks (`has_<sys>_cluster`).
- [`configs/configs_mds_signal/`](configs/configs_mds_signal/) — all truth fills
  and the efficiency numerator/denominator pairs (LLP η is signed, 60 bins
  −3..3, not \|η\|); the object/event masks (fiducial, ≥10 hits, matched
  cluster), written once per system under `_repeat`; and the derived plots —
  the factorized efficiency chain with detector-station bands, matched
  fractions, sig-vs-bkg overlays.
- [`configs/configs_mds_gen/`](configs/configs_mds_gen/) — the gen-level-only
  subset (LLPs + matched rechits); its `columns.yaml` drops the DBSCAN steps.
- [`custom/`](custom/) — `derive()` and its helpers: [`params.py`](custom/params.py)
  (`PARAM_SPEC`, the `columns.yaml` schema), [`clustering.py`](custom/clustering.py)
  (DBSCAN), [`llp.py`](custom/llp.py) (`events.llp` and the per-LLP rechit
  spread), [`helpers.py`](custom/helpers.py) (object selection, isolation ΔR,
  the jet ID). The prompt-object cuts themselves are ordinary selections in
  [`configs/_common/prompt_objects.yaml`](configs/_common/prompt_objects.yaml),
  shared by the three sets that are overlaid.
  Everything stays reachable through `custom.columns`. See
  [`docs/derived-columns.md`](docs/derived-columns.md).

---

### Event display (r–z rechit picture of the clusters)

[`scripts/event_display.py`](scripts/event_display.py) draws one figure per
event with every CSC/DT rechit at its global (|z|, r), coloured by DBSCAN
cluster — the same clustering `derive()` uses (`-c <config set>` reads that
set's `columns.yaml`, so the display clusters exactly as its histograms were
filled). One colour per cluster (CSC and
DT are told apart by position, not by colour or marker) and the marker is the
hit-level truth: `o` for a rechit carrying an `llpIdx`, `x` for one that does
not, so the LLP shower and the activity DBSCAN swept up with it stay visible
separately. Noise hits are light grey dots, the grey boxes are the ME/MB
chambers and the solenoid (`drawRZ()`, DT drawn from |z| = 0), and an open star
marks the truth shower position (`llpSim*`) of each truth-matched cluster.
RPC is not drawn. The chamber layout is a z > 0 quarter view, so the display is
folded to |z| and a −z cluster lands on the same picture as a +z one.

```bash
conda activate mds
export X509_USER_PROXY=$HOME/private/.proxy
python scripts/event_display.py -c configs/configs_mds -d suep_mDark2_temp1 -n 5 --matched-only --min-size 50 --zoom --with-etaphi
```

Figures land in `/groups/hephy/cms/ang.li/suep_plots/event_display/` as
`evd_<dataset>_<file>_ev<entry>.png`. Useful flags: `--zoom` crops to the
clustered hits (a shower is a few tens of cm across, the full view is mostly
empty), `--with-etaphi` adds the η–φ panel the clustering actually runs in,
`--matched-only` / `--min-size` pick interesting events, `--systems csc`
restricts to one system, and `-f <file.root> --entries 3 7` draws specific
entries of a specific file. It
works on background/data too (`-d dy_2mu_50to120`, `-d zerobias_2024C`); those
have no truth branches, so every cluster is labelled "unmatched".

---

### Gen-level rechit spread of one LLP

These plots live only in `configs/configs_mds_gen/`, a trimmed, self-contained config —
LLP gen kinematics, per-LLP matched-rechit counts, the ΔR plots below, and 2D
maps of CSC nHits and ΔR₉₀ against |η| / pT / energy / boost (sections `4d`–`4f`;
their `profile_x` curves are in `derived_plots.yaml`), and nothing else (68
histograms against 237, signal samples only, since the background has no
`SUEPGenPart`). It touches no cluster collection, so its
[`columns.yaml`](configs/configs_mds_gen/columns.yaml) enables only the
`llp`/`llp_hits`/`llp_shape` steps and skips DBSCAN (~35 % of `derive()`).
`events.<sys>Cluster` and `llp.reco*` are then deliberately *not* attached, so a
config that needs them fails at expression validation instead of quietly
filling empty histograms. `configs/configs_mds/` + `configs/configs_mds_signal/` keep the full
reconstruction study (clusters, efficiencies) and do not repeat these
rechit-spread plots.

The `4d`/`4e` maps show that the CSC is an endcap: both the hit count and ΔR₉₀
turn on at |η| ≈ 1, plateau across the endcap, and fall off past |η| ≈ 2.4, in
lockstep — so |η| drives both axes of `llp_dr90_vs_nhits_csc`. At fixed nHits,
ΔR₉₀ is larger at high |η| (the η-φ metric stretches toward the beamline). nHits
saturates with LLP energy (~40 hits) while ΔR₉₀ peaks near 20 GeV and then
declines, the high-energy compact-shower regime behind the turnover in the
nHits map.

The `4f` maps add the boost. The gen `llp.openingAngle` (3D angle between the
two decay daughters) falls as a clean 1/βγ curve — a more boosted LLP decays
into a tighter pair. But ΔR₉₀ vs opening angle is **non-monotonic**: it peaks
near 0.3 rad and falls off on both sides. Toward small angle (high boost) the
two daughter showers merge and ΔR₉₀ drops to the single-shower floor (~0.08 at
these |η|) — the boost/collimation effect. Toward wide angle (low boost) ΔR₉₀
also drops, because the rechits are then dominated by a single daughter (the
other leaves the CSC acceptance or too few hits). The controlled
`llp_dr90_vs_betagamma_ctrl` (fixed 1.6<|η|<2.2, nHits<25) isolates the boost
side: past the βγ≈5 peak, ΔR₉₀ falls with boost as expected. So a raw
`dR₉₀ vs βγ` profile is confounded (|η| and nHits both rise with βγ); the
collimation shows only once those are pinned.

These answer "how wide is the rechit shower of a single LLP", from truth only
(`cscRechits_llpIdx` etc.), with no clustering involved. Per LLP, over the
rechits carrying its `llpIdx`:

| plot | field | meaning |
| --- | --- | --- |
| `llp_nhits_<sys>` | `nHits<SYS>` | matched rechits per LLP |
| `llp_drpair_min/max_<sys>` | `drPairMin/Max<SYS>` | smallest / largest ΔR between two of its rechits |
| `llp_dr{50,80,90}_<sys>`, `llp_drmax_<sys>` | `dr50/dr80/dr90/drMax<SYS>` | radius around the rechit centroid holding 50/80/90/100 % of them |
| `<sys>_rechit_dr_llp` | `<coll>.drLLP` | hit-weighted: every matched rechit's ΔR to its LLP's centroid |

`<sys>` is `csc`, `dt`, `rpc` or `total` (the three pooled). The `_hits10`
variants keep only LLPs with ≥10 matched rechits — the number a DBSCAN cluster
needs to be truth-matched, and the regime where a cone size is meaningful.
LLPs with <2 matched rechits have no ΔR and drop out of the fill.

Results on `suep_temp1` (500k events, median over LLPs with the [16 %, 84 %]
band; DBSCAN currently runs with eps = 0.2). `suep_temp2` agrees to within
≈0.01 everywhere, so the numbers barely depend on the SUEP temperature:

| system | ΔR(50 %) | ΔR(80 %) | ΔR(90 %) |
| --- | --- | --- | --- |
| CSC | 0.031 [0.006, 0.105] | 0.042 [0.007, 0.151] | 0.048 [0.008, 0.185] |
| CSC, ≥10 hits | 0.048 [0.017, 0.127] | 0.067 [0.023, 0.185] | 0.077 [0.025, 0.229] |
| DT | 0.074 [0.023, 0.124] | 0.113 [0.032, 0.189] | 0.129 [0.038, 0.222] |
| RPC | 0.053 [0.024, 0.125] | 0.067 [0.027, 0.165] | 0.072 [0.027, 0.188] |

Hit-weighted (pooling all matched rechits instead of averaging over LLPs) the
containment radii are 0.053 / 0.134 / 0.205 for CSC and 0.088 / 0.163 / 0.226
for DT at 50 / 80 / 90 %. The closest pair of rechits of one LLP sits at
ΔR ≈ 1e-3 (CSC/DT strip granularity) and the farthest at ΔR ≈ 0.1 (CSC) to
0.22 (DT).

---

## Helper scripts

Two of these are about the framework rather than the physics:

- **`bash scripts/smoke_test.sh [outdir]`** — processes one signal file through
  [`configs/configs_smoke/`](configs/configs_smoke/) (one histogram, no truth,
  no isolation) and plots it. Run it after changing anything in `src/` or
  `custom/`: it exercises run → pickle → plot in about a minute instead of the
  several a real config set takes. It is not a physics check.
- **`python scripts/dump_config.py <config set>...`** — prints a config set's
  *fully expanded* definitions (after `_extends` / `_repeat` / `_include` /
  `_registry`) as canonical YAML. Dump before a change to the config layer and
  `diff` after: an empty diff proves the expansion is unchanged without
  reprocessing a single ROOT file.


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
`<var>_matched` histograms come from a `configs/configs_mds_signal` run, the background's
inclusive `<var>` from a `configs/configs_mds` run; both configs define the inclusive
histograms identically, so the axes match and nothing needs refilling. Also
writes `separation.txt` (total-variation distance per variable):

```bash
python scripts/compare_sig_bkg.py output_mds_grid output_mds_data sigbkg_dir \
    --suffix _matched -c configs/configs_mds_signal
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
[the MDS study section below](#event-display-rz-rechit-picture-of-the-clusters)
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

## Environment knobs

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

A variation on an existing config set is an `_extends` plus the one line that
differs — never a copy, which drifts the moment someone adds a plot to one of
the two:

```bash
mkdir -p configs/configs_mds_minpts50
for f in samples histograms selections corrections reweights derived_plots; do
    printf '_extends: ../configs_mds\n' > "configs/configs_mds_minpts50/$f.yaml"
done
cat > configs/configs_mds_minpts50/columns.yaml <<'EOF'
_extends: ../configs_mds
parameters:
  cluster_min_samples: 50
EOF
suep-run -c configs/configs_mds_minpts50 -o output_mds_minpts50 --chunk-size 10000
```

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

**Add a sample** — append to `samples.yaml` (see [samples.yaml](docs/configuration.md#samplesyaml)).

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
[reweights.yaml](docs/configuration.md#reweightsyaml): expression- or map-based, event- or
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
topic over `custom/params.py`, `clustering.py`, `llp.py` and `helpers.py`, all
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
| `ERROR in columns.yaml: ...` | Unknown key/parameter/step, or a step whose dependency is not enabled — see the table in [docs/derived-columns.md](docs/derived-columns.md#steps). Fatal by design: a typo here would mean silently different histograms. |
| Merged output looks inconsistent | Shards filled with different derived-column settings. Keep the knobs in the config set's `columns.yaml` (not in `$MDS_*`) and one output directory per (config set × clustering). |
| Plots have no labels / wrong colors | `suep-plot` was called without `-c <config set>`, so styling fell back to defaults. |
| Empty or truncated pickles after a Slurm run | `suep-status -o <dir>` finds them; `--resubmit` reruns exactly those tasks. |
| Efficiency plots look distorted | `--lumi` was passed to an MDS config set (placeholder `xs: 1.0`). Drop it. |
| `ImportError: _lazywhere` from `coffea.lookup_tools` | `suep_plot` was bypassed — import `suep_plot` (or the shim) first; see [the compat shim](#corrections--the-scipy-compat-shim). |
