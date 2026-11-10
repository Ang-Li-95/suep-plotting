# suep-plotting

A config-driven, [coffea](https://github.com/CoffeaTeam/coffea)-based histogram
framework for the SUEP MDSNano analysis: NanoEvents in, CMS-style
([mplhep](https://github.com/scikit-hep/mplhep)) figures out. What to plot is
YAML, not Python:

- **variables** are written once, in a catalog (`variables.yaml`): expression,
  binning, axis title;
- **histograms** are organized as regions (`histograms.yaml`): event-level cuts
  → object-level cuts → the variables to fill after both;
- a **config set** (`configs/<set>/`) is one study, inheriting from shared bases.

The muon-system objects (LLPs, DBSCAN clusters of the CSC/DT/RPC rechits) are
built by `custom/` -- see [docs/derived-columns.md](docs/derived-columns.md).

| Read | for |
|---|---|
| this README | setup, a first run, adding a plot |
| [docs/running.md](docs/running.md) | local runs, Slurm, following and resubmitting, plotting, troubleshooting |
| [docs/configuration.md](docs/configuration.md) | every config file: the catalog, regions, selections, samples, derived plots, errors |
| [docs/derived-columns.md](docs/derived-columns.md) | `custom/`: the LLP and cluster collections, `columns.yaml` |
| [docs/mds-study.md](docs/mds-study.md) | the MDS cluster study: RPC modes, timing, the six cluster populations, event display |
| [docs/known-issues.md](docs/known-issues.md) | clustering memory on dense Cosmics events |
| [docs/design-notes.md](docs/design-notes.md) | for readers of the code: which module does what, and why it is built this way |
| [docs/histogram-renames.tsv](docs/histogram-renames.tsv) | old histogram name → new `region/cut/variable` name, per set |

---

## Setup

```bash
git clone <this repo> && cd suep-plotting
conda env create -f environment.yml          # once; on CBE.next the shared env is `rootmds`
eval "$(conda shell.bash hook)" && conda activate rootmds
pip install -e .                             # editable: code and config edits apply at once
```

This installs six commands -- `suep-run`, `suep-plot`, `suep-submit`,
`suep-status`, `suep-reweight`, `suep-config` -- each with `-h` and `--version`.
Without the install they work as `python -m suep_plot.cli <command>`.

Most inputs are `root://` URLs, which need a grid proxy on a shared filesystem
(so batch jobs can read it too):

```bash
export X509_USER_PROXY=$HOME/private/.proxy
voms-proxy-info -exists -valid 2:00 || \
  voms-proxy-init -voms cms --valid 192:00 --vomslife 192:0 -out $X509_USER_PROXY
```

The commands check it before reading anything and say how to renew it.

---

## A first run

```bash
suep-config check configs/configs_mds          # does the set work? (seconds)
suep-run  -c configs/configs_mds -o $SCRATCHDIR/try \
          -s suep_mDark2_temp1 --file-range 0:1 --part 0     # one file, locally
suep-plot $SCRATCHDIR/try                       # -> $SCRATCHDIR/try/plots/index.html
```

`-c` names one config set; a wrong name is answered with the list of sets.
`suep-plot` finds the config set and the figure folder from the run's
`run_info.json`. The figures sit in folders per region and cut, and
`index.html` has a filter box.

Anything beyond a few files goes to Slurm -- the login node is for editing and
submitting:

```bash
OUT=/groups/<group>/<user>/suep_output                # keep runs on group storage
suep-submit -c configs/configs_mds -o $OUT/configs_mds --files-per-job 10
suep-status -o $OUT/configs_mds                       # now and then: done / queued / failed + logs
suep-status -o $OUT/configs_mds --resubmit --mem 16000   # redo the failed tasks, with more memory
suep-plot   $OUT/configs_mds                          # once complete
```

`suep-submit` picks the QOS from `--time`, freezes each task's input files,
and refuses to submit over an existing run. `suep-status` shows each unfinished
task's Slurm state and log. `suep-plot` sums exactly the pickles this run's
tasks wrote. [docs/running.md](docs/running.md) has the details.

---

## Adding a plot

1. **The variable.** Look for it in [`variables.yaml`](variables.yaml)
   (`suep-config show <set> --section variables`); if it is not there, add a line:

   ```yaml
   csc_cluster_size_over_r: {expression: "events.cscCluster.size / events.cscCluster.r", bins: [50, 0, 2], label: 'CSC cluster $N_{\mathrm{hits}}/r$ [1/cm]'}
   ```

   To see which collections and fields there are:
   `suep-config fields configs/configs_mds -s suep_mDark2_temp1 --grep cscCluster --ranges`.

2. **Where to fill it.** List it under a region and cut in the set's
   `histograms.yaml` -- or in a base, to have it in every study built on it:

   ```yaml
   regions:
     csc:
       objects:
         all:      {vars: [..., csc_cluster_size_over_r]}
         iso_muon: {cut: csc_cluster_iso_muon, label: muon-isolated, vars: [csc_cluster_size_over_r]}
   ```

   This fills `csc/all/csc_cluster_size_over_r` and
   `csc/iso_muon/csc_cluster_size_over_r`. A different binning or title for one
   histogram: `{var: csc_cluster_size_over_r, bins: [100, 0, 4]}`. A 2D
   histogram: `[x_variable, y_variable]`.

3. **A new cut** is a line in `selections.yaml`
   (`csc_cluster_big: "events.cscCluster.size > 100"`) and a new entry under
   `objects:`. Whether a cut or a variable is per object or per event is read
   off its expression.

4. **Check, then run**: `suep-config check <set>`, then a one-file
   `suep-run` as above. Labels, colours and log axes can change afterwards
   without reprocessing; `suep-plot` re-reads them.

Efficiencies and ratios between two cuts of a region go in
`derived_plots.yaml`, one entry for a list of variables:

```yaml
matchfrac_csc:
  type: efficiency
  region: csc
  numerator: matched
  denominator: all
  vars: [csc_cluster_size, csc_cluster_eta]
```

A study that differs from an existing one in a setting is two files, not a
copy -- see `configs/configs_mds_minpts50/`:

```yaml
# set.yaml
extends: ../configs_mds
description: "configs_mds with the standard MDS clustering: min cluster size 50"
# columns.yaml
_extends: ../configs_mds
parameters: {cluster_min_samples: 50}
```

Mistakes stop the run with a message naming the file, histogram, sample and
expression, before anything is processed -- nothing fills empty without a word.

---

## Config sets in this repo

`suep-config show configs/<set>` lists what a set fills and where each
definition comes from.

| Set | What it fills |
|---|---|
| `common`, `common_gen`, `common_clustersrc` | **bases** (not run on their own): reco cluster plots; + LLPs, truth cuts and efficiencies; + one cluster population per set |
| `configs_mds` | reco-only CSC/DT cluster properties and shower shapes (2 signal points + DY) |
| `configs_mds_minpts50` | `configs_mds` with the standard MDS min cluster size 50 |
| `configs_mds_signal` | truth level over the Gen3 signal grid: LLPs, matched/unmatched clusters, efficiency chain |
| `configs_mds_gen` | `configs_mds_signal` plus the LLP shower-shape family |
| `configs_mds_data` | reco cluster plots with isolation cuts, on signal, DY and ZeroBias data |
| `configs_mds_cosmics` | reco cluster plots on Cosmics (no prompt objects) |
| `configs_mds_rpcmerge`, `_rpcmerge_data` | the signal / data studies with `rpc_mode: merge` |
| `configs_mds_src_*` (six) | where a cluster comes from: LLP, DY in a jet / on a muon / isolated, ZeroBias, Cosmics |
| `configs_mds_objects` | prompt objects (MET, jets, leptons) and cluster multiplicities |
| `configs_mds_trigger` | HLT/L1 MDS trigger efficiencies |
| `configs_mds_sigonly` | `configs_mds` on the two signal points only |
| `configs_g4compare` | pre/post Geant4-fix signal (its inputs are currently gone from EOS) |
| `configs_smoke` | the smoke test: one file, two histograms |

**Never pass `--lumi` to the MDS sets**: their `xs: 1.0` is a placeholder.
[docs/mds-study.md](docs/mds-study.md) describes the study and how to
reproduce it (`bash reproduce.sh <output-root>` submits the three main runs).

---

## Repository layout

```
suep-plotting/
├── variables.yaml        # the variable catalog
├── datasets.yaml         # the dataset registry (files, labels, colours)
├── configs/<set>/        # one directory per config set (see above)
├── custom/               # derive(): LLPs, DBSCAN clustering, isolation, jet ID
├── src/suep_plot/        # the framework, one module per job (docs/design-notes.md)
├── scripts/              # comparison plots, event display, dump_config, smoke test
├── studies/              # written-up one-off studies (own READMEs)
├── docs/                 # the reference documentation
├── tests/                # pytest, no input files needed
├── environment.yml       # the conda environment
└── reproduce.sh          # submits the MDS study runs
```

## Developing

```bash
srun --partition=c --qos=short --time=00:30:00 --mem=8000 --cpus-per-task=4 \
     bash -c 'eval "$(conda shell.bash hook)" && conda activate rootmds && python -m pytest -q'
bash scripts/smoke_test.sh                       # run -> pickle -> plot on one file
python scripts/dump_config.py configs/*/ -o <dir>   # expanded configs, to diff before/after
```

See [docs/running.md](docs/running.md#checking-a-change-to-the-framework).
