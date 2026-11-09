---
name: run-suep
description: Run, submit and plot with the suep-plotting framework — suep-run / suep-submit / suep-status / suep-plot / suep-config check — plus the environment and grid proxy they need, which config set and output directory to use, and the traps (local runs get killed, `-c` must name a config set, never pass --lumi to the MDS sets). Use this whenever the user asks to process samples, fill histograms, re-plot, submit to Slurm, check or resubmit failed jobs, or asks "why did the run fail / produce nothing". For *editing* configs (new histogram, selection, efficiency plot), use the add-plots skill instead.
---

# Running, submitting and plotting

Run every command from the root of the suep-plotting checkout. Config-set
paths (`configs/...`) are relative to it. `docs/running.md` is the full
reference; this skill is the short version, plus the traps.

## 1. Environment: do this first, in every shell

```bash
eval "$(conda shell.bash hook)" && conda activate rootmds
export X509_USER_PROXY=$HOME/private/.proxy
voms-proxy-info -exists -valid 2:00 || \
  voms-proxy-init -voms cms --valid 192:00 --vomslife 192:0 -out $X509_USER_PROXY
```

- **The conda env:**
  - `rootmds` is the shared env on CBE.next (coffea 2026.x, correctionlib,
    scikit-learn for DBSCAN, xrootd).
  - Elsewhere, build it with
    `conda env create -f environment.yml && pip install -e .`.
- **The grid proxy:**
  - Nearly every sample is `root://eos.grid.vbc.ac.at//...`, so the proxy is
    mandatory.
  - The commands check it before reading anything and refuse with the renewal
    command.
  - Keep it on a shared filesystem so batch jobs can read it.
  - Never force `XrdSecPROTOCOL`: the gsi proxy is the only authentication
    that works.
- **cvmfs** must be mounted for any set with `jerc: true`. The JEC/JER and
  jet-ID payloads come from `/cvmfs/cms-griddata.cern.ch/cat/metadata/JME/`.
- **A command missing from `PATH`** (`suep-config`, `suep-status`) means the
  installed entry points are older than the code.
  - `python -m suep_plot.cli {run,plot,submit,status,reweight,config} ...`
    always works.
  - `pip install -e .` regenerates the entry points. **Ask the user first**,
    because the env is shared.

## 2. `-c` names one config set; `-o` names one run

```bash
suep-run -c configs/configs_mds ...     # right
suep-run -c configs ...                 # ERROR: lists the sets with their descriptions
```

- `configs/common`, `common_gen` and `common_clustersrc` are **bases**
  (`abstract: true` in their `set.yaml`), and `suep-run` / `suep-submit`
  refuse them.
- To see what a set fills, run `suep-config show configs/<set>`. The README
  and the add-plots skill have the table of sets.

**One output directory per run** (config set × clustering), never reused:

- `suep-submit` refuses a directory that already holds a run. `--overwrite`
  moves its `slurm/` aside to `slurm.prev-<time>` and treats the old pickles
  as stale.
- `run_info.json` in the run directory records the config set, the arguments
  and the git commit. `suep-plot <run>` reads `-c` from it.
- Keep real runs on group storage, e.g.
  `/groups/<group>/<user>/suep_output/<config>_<gen>_<clustering>/`.
  `$SCRATCHDIR` is deleted after 28 days.

## 3. Check before you run

```bash
suep-config check configs/configs_mds            # references, columns.yaml, every expression on one file per sample
suep-config check --static configs/configs_mds   # without opening files (no proxy needed)
```

- Every expression of every histogram and selection is evaluated on each
  sample's first file. Any failure is **fatal** and names the histogram, the
  sample and the expression. `suep-run` does the same check at startup.
- **A failure is real.** For example, a gen expression in a set without the
  gen steps, or a branch a sample lacks. Fix the config or enable the step;
  don't work around it.
- One set is fine on the login node. Run many sets through
  `srun --partition=c --qos=rapid --time=00:30:00 --mem=8000 suep-config check configs/*/`.

## 4. Local runs: small only

```bash
suep-run  -c configs/configs_mds -o $SCRATCHDIR/try -s suep_mDark2_temp1 --file-range 0:1 --part 0
suep-plot $SCRATCHDIR/try                      # -> $SCRATCHDIR/try/plots/index.html
```

- **Do not run a full sample or a production on the login node.** It has a
  process cap; many-worker runs get killed (`fork: Resource temporarily
  unavailable`). One file is fine; anything more goes to Slurm (§5).
- `--file-range` processes part of a sample and therefore needs a `--part`
  tag.
- `--chunk-size` defaults to 10000. It changes memory use, not the result.
- **When a sample is skipped:**
  - A sample is skipped when its pickle is newer than every config file it
    reads, the `custom/` modules and its *local* inputs.
  - `root://` inputs cannot be dated, so they are always reprocessed.
  - Changes to `src/suep_plot/` are not tracked: pass `-f/--force`.

## 5. Slurm: the real path

```bash
OUT=/groups/<group>/<user>/suep_output
suep-submit -c configs/configs_mds -o $OUT/configs_mds --files-per-job 10 --dry-run   # read slurm/task_list.txt + job.sh
suep-submit -c configs/configs_mds -o $OUT/configs_mds --files-per-job 10
```

| option | default | notes |
|---|---|---|
| `--time` | `04:00:00` | Slurm format: `MM:SS` is minutes, `HH:MM:SS`, `D-HH:MM` ... |
| `--qos` | from `--time` | `short` (the cluster default) up to 8 h, `medium` up to 2 d, `long` up to 14 d. A QOS that is too short for `--time` is refused before `sbatch` |
| `--mem` | `8000` | MB, or `16G` |
| `--files-per-job` | the dataset's `files_per_job`, else one task per sample | an array index above 3000 is refused, with a size that fits |
| `--conda-env` | the active env | what the tasks activate |
| `-s` | all | samples, space- or comma-separated; an unknown name lists the valid ones |
| `--proxy` | `$X509_USER_PROXY`, else `~/private/.proxy` | baked into `job.sh` |
| `--overwrite` | off | submit into a directory that already holds a run |

**What it does:**

- Input files are resolved **once, at submission**, into
  `slurm/filelists/`. A dataset that grows later cannot shift the shards, and
  a resubmitted task reruns exactly its own files.
- Job ids go to `slurm/jobs.tsv`. A fingerprint of every config file and
  `custom/` module goes to `slurm/fingerprint.json`.
- Logs go to `slurm/logs/<job name>_<job id>_<index>.out/.err`. The job name
  is `suep_<run directory>`.
- The array has no `%N` throttle: tasks run as wide as the QOS allows.
- `suep-submit` warns when the proxy expires before `--time`.
- Mail does not work on CBE.next; follow the run with `suep-status`.

**Sizing:** measure one file first. Signal MDSNano files are 20k events, and
`derive()` with the reco steps runs at about 60 events/s. That is about
5.5 min per file on one worker, so `--files-per-job 5` is about 30 min per
task. Cosmics files are dense and ZeroBias files are small; their datasets
carry their own `files_per_job`.

**Pre-flight for a real submission:**

1. Check the proxy (§1).
2. Check that cvmfs is reachable if `jerc: true`.
3. Run `suep-config check` (§3).
4. Run `--dry-run`, then read `task_list.txt` and `job.sh`.
5. Submit.

## 6. Follow, then redo what failed

```bash
suep-status -o $OUT/configs_mds                       # done / pending / running / failed, with log paths
suep-status -o $OUT/configs_mds --resubmit --dry-run  # what would be redone
suep-status -o $OUT/configs_mds --resubmit --mem 16000         # OUT_OF_MEMORY
suep-status -o $OUT/configs_mds --resubmit --time 12:00:00     # TIMEOUT; QOS follows --time
```

**`suep-status`:**

- A task is **done** when its pickle exists, reads back (`--no-verify` skips
  that), and was written by this submission. An older pickle is *stale*.
- Unfinished tasks are grouped by their Slurm state, from one `sacct` call,
  with each log path. Pickles that no task of the run writes are listed;
  `suep-plot` ignores them.
- Run it now and then, **not in a loop**: every call queries the shared
  scheduler. At most every few minutes.

**`--resubmit`:**

- It runs `sbatch --array=<failed indices>` on the original `job.sh` and
  leaves queued and running tasks alone.
- `--time` / `--mem` / `--qos` / `--partition` override the script.
- **It refuses in these cases:**
  - a `job.sh` that can no longer run: it calls the removed `cli_worker`, or
    its config set or file lists are gone;
  - a config file or `custom/` module that changed since submission, because
    the redone tasks would fill differently. `--allow-config-change` is only
    for edits that cannot change the fills, such as labels or comments. Ask
    the user before using it.

**Exit code:** 0 when everything is done, queued or resubmitted; 1 otherwise.

## 7. Plot

```bash
suep-plot $OUT/configs_mds                        # config set from run_info.json, figures to $OUT/configs_mds/plots
suep-plot $OUT/configs_mds --formats png -j 4     # faster while iterating
suep-plot $OUT/run_a $OUT/run_b -o $OUT/cmp -c configs/configs_mds   # overlay the samples of two runs
```

- **What is summed:**
  - It sums the pickles **this run's tasks wrote**, and the parts of a
    sharded sample become one sample.
  - Missing, stale or foreign pickles are reported, never silently summed.
  - A directory holding both `X.pkl` and `X.part<k>.pkl` is refused, because
    that would count events twice.
- **What it writes:**
  - Figures go into `plots/<region>/<cut>/<variable>.png`.
  - `index.html` groups the figures by folder, has a filter box, and shows
    only this run's figures.
  - `cutflow.txt`/`.csv` count the events passing each selection, per
    sample.
  - `--save-root merged.root` exports every histogram as
    `<histogram>/<sample>`.
- **Styling** (labels, colours, `rebin`, `log_*`) and **every derived plot**
  are re-read from the config set on each call. Changing them needs no
  reprocessing.
- **Exit code** 1 if no pickle was found or any figure failed.

`bash $OUT/configs_mds/slurm/merge_and_plot.sh` is the same as `suep-plot $OUT/configs_mds`.

## 8. Traps that have bitten

- **Never pass `--lumi` to an MDS set.** `xs: 1.0` there is a placeholder
  (`suep-plot` warns), and lumi scaling distorts the Clopper–Pearson
  efficiency intervals.
- **Runs filled before the region conversion (2026-11-09)** have the old flat
  histogram names (`csc_cluster_size_iso_muon`, not
  `csc/iso_muon/csc_cluster_size`).
  - Today's config set cannot restyle them, because their names are not in
    it.
  - Its derived plots warn `source histogram(s) ... not found` and are
    skipped.
  - Refill the run, or look the names up in `docs/histogram-renames.tsv`.
- **An old run without `run_info.json`:** pass `-c <set>` to `suep-plot`.
  Otherwise there are no labels, colours or derived plots, and the figures go
  to `./plots`.
- **`xrootd ... [3010] Operation not permitted` inside a job** means the job
  has no valid proxy.
  - `job.sh` exports `X509_USER_PROXY="${X509_USER_PROXY:-<path>}"`, so a
    proxy exported in the submitting shell wins.
  - `--resubmit` checks that proxy first.
- **Killed for memory:** use `--resubmit --mem 16000`. DBSCAN memory grows
  with the rechits of one event (dense Cosmics events; see
  `docs/known-issues.md`), so a lower `--chunk-size` helps less than more
  `--mem`.
- **`TIMEOUT` on some tasks but not others:** the shard is big. Resubmit with
  a longer `--time`; the QOS follows. Next time, use a smaller
  `--files-per-job`.
- **"up to date, skipping"** happens only for local inputs. Pass `--force`
  after changing `src/`.
- **A JEC tag missing from the payload:** the tag in `suep_plot.jme.DEFAULTS`
  is not in the reachable `jet_jerc.json.gz`. The error lists what is
  available.

## 9. Helper scripts (plot time, on existing pickles)

```bash
python scripts/compare_eps.py $OUT/configs_mds $OUT/configs_mds_minpts50 $OUT/cmp_minpts "min 10" "min 50"
python scripts/compare_sig_bkg.py $OUT/configs_mds_signal $OUT/configs_mds_data $OUT/sigbkg \
    --sig-cut matched --bkg-cut all -c configs/configs_mds_signal
python scripts/event_display.py -c configs/configs_mds -d suep_mDark2_temp1 -n 5 \
    --matched-only --min-size 50 --zoom --with-etaphi
```

- `compare_eps.py` draws each histogram of each sample from both runs,
  unit-normalized and with their ratio, ranked by how much the shape changed.
- `compare_sig_bkg.py` pairs `<region>/<sig-cut>/<var>` of the signal run with
  `<region>/<bkg-cut>/<var>` of one or more background runs, and ranks the
  variables by separation (`separation.txt`).
- `event_display.py` clusters through `custom/clustering.py`, so it follows
  the set's `rpc_mode`. It refuses `merge`.

To reproduce the MDS study, run `bash reproduce.sh <output-root> [--dry-run]`.
It submits `configs_mds`, `configs_mds_minpts50` and `configs_mds_signal`.

## Deep dives (read only when needed)

- `docs/running.md`: every option, the run directory layout,
  troubleshooting, and how to check a framework change (tests via `srun`,
  `scripts/smoke_test.sh`, `scripts/dump_config.py`).
- `docs/configuration.md`: every YAML file and directive.
- `docs/derived-columns.md`: `custom/`, with the collections, parameters and
  steps.
- `docs/mds-study.md`: the MDS cluster study and its reproduction.
- `docs/known-issues.md`: clustering memory grows as N² per event.
- The add-plots skill: editing histograms, selections and derived plots.
