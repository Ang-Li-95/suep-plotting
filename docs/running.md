# Running, submitting and plotting

The workflow is: fill histograms (locally for a quick check, on Slurm for
anything real), follow the Slurm run, then plot. Everything below assumes the
environment is active and you are in the repo root:

```bash
eval "$(conda shell.bash hook)" && conda activate rootmds
export X509_USER_PROXY=$HOME/private/.proxy      # every root:// read needs it
```

`rootmds` is the shared environment on CBE.next; `environment.yml` rebuilds it
anywhere (`conda env create -f environment.yml && pip install -e .`).
The proxy lives on a shared filesystem so batch jobs can read it; create or
renew it with

```bash
voms-proxy-init -voms cms --valid 192:00 --vomslife 192:0 -out $HOME/private/.proxy
```

Every command checks the proxy before it reads a `root://` input and says how to
renew it when it is missing or about to expire.

---

## A quick local check

The login node is for editing and submitting, not computing: a few events of
one file is fine, a sample is not.

```bash
suep-config check configs/configs_mds                # seconds: references + every expression
suep-run  -c configs/configs_mds -o $SCRATCHDIR/try \
          -s suep_mDark2_temp1 --file-range 0:1 --part 0
suep-plot $SCRATCHDIR/try                             # -> $SCRATCHDIR/try/plots/index.html
```

- `-c` names one config set (a directory under `configs/`); a wrong name is
  answered with the list of sets and their descriptions.
- `--file-range` processes part of a sample, so it needs a `--part` tag.
- `suep-run` writes one pickle per sample (`<sample>.pkl`, or
  `<sample>.part<k>.pkl` for a shard) and a `run_info.json` recording the
  config set, the arguments and the git commit.
- A sample is skipped when its pickle is newer than the config files, the
  `custom/` modules and its *local* input files; `root://` inputs cannot be
  checked, so they are always reprocessed. Code changes in `src/` are not
  tracked: pass `--force`.
- `--chunk-size` (default 10000) does not change the result, only memory.

For anything bigger use Slurm. `srun` gives you a compute node for a short
interactive check:

```bash
srun --partition=c --qos=rapid --time=00:30:00 --mem=8000 --cpus-per-task=2 \
     bash -c 'eval "$(conda shell.bash hook)" && conda activate rootmds && suep-config check configs/*/'
```

---

## On Slurm

### Submitting

```bash
OUT=/groups/<group>/<user>/suep_output            # keep runs on group storage
suep-submit -c configs/configs_mds -o $OUT/configs_mds --files-per-job 10 --dry-run
suep-submit -c configs/configs_mds -o $OUT/configs_mds --files-per-job 10
```

One array task per sample, or per `--files-per-job` shard of one. A dataset
can carry its own shard size (`files_per_job:` in `datasets.yaml`), used when
the flag is not given -- Cosmics files are dense, ZeroBias files are small.

| Option | Default | Notes |
|---|---|---|
| `--time` | `04:00:00` | Slurm's own format: `MM:SS` is minutes, `HH:MM:SS`, `D-HH:MM` ... |
| `--qos` | from `--time` | none up to 8 h (`short`), `medium` up to 2 days, `long` up to 14 days |
| `--mem` | `8000` | MB, or `16G` |
| `--files-per-job` | the dataset's, else one task per sample | an array index above 3000 is refused, with the size that fits |
| `--conda-env` | the active one | what the tasks activate |
| `--chunk-size` | `10000` | events per chunk |
| `-s` | all | samples, space- or comma-separated |
| `--overwrite` | off | submit into a directory that already holds a run (see below) |

Sample files are resolved **once, at submission**: each task's files are
written to `slurm/filelists/` and the task reads that list, so a dataset that
grows later cannot shift the shards, and a resubmission reruns exactly the
same inputs.

A run directory holds one run. Submitting into a directory that already has a
submitted run is refused; `--overwrite` moves its `slurm/` aside to
`slurm.prev-<time>`, and pickles left from it are then ignored as stale.

What `suep-submit` writes:

```
<output>/
├── run_info.json           # config set, arguments, git commit, host
└── slurm/
    ├── job.sh              # the array script (proxy path baked in)
    ├── task_list.txt       # one line per array index: sample, file list, part
    ├── filelists/          # the frozen input files of each task
    ├── jobs.tsv            # every sbatch of this run: job id, array, time
    ├── fingerprint.json    # sha256 of the configs and custom/ at submission
    ├── logs/               # <job name>_<job id>_<index>.out / .err
    └── merge_and_plot.sh   # suep-plot over the finished run
```

The job name is `suep_<run directory>`, so runs are told apart in `squeue`.
Slurm mail does not work on CBE.next; follow the run with `suep-status`.

### Following a run, and redoing what failed

```bash
suep-status -o $OUT/configs_mds                        # what is done, queued, failed
suep-status -o $OUT/configs_mds --resubmit             # redo the failed tasks
suep-status -o $OUT/configs_mds --resubmit --time 16:00:00 --mem 16000
```

`suep-status` compares the pickles with `task_list.txt` and asks Slurm (one
`sacct` call) about the rest:

- a task is **done** when its pickle exists, reads back, and was written by
  this submission (an older pickle in the directory is *stale*);
- unfinished tasks are grouped by Slurm state -- PENDING / RUNNING, or
  FAILED / TIMEOUT / OUT_OF_MEMORY with the path of their log;
- pickles that no task of the run writes are listed (suep-plot ignores them).

`--resubmit` sends `sbatch --array=<failed indices>` over the original
`job.sh`, so the redone tasks own exactly the same input files. It

- leaves tasks that are still queued or running alone;
- puts `--time` / `--mem` / `--qos` / `--partition` on the sbatch line, where
  they override the script (a task that hit its limit needs more of it), and
  picks the QOS a longer `--time` needs;
- refuses a `job.sh` that can no longer run (it calls the removed
  `cli_worker`, its config set or file lists are gone), and refuses when a
  config file or a `custom/` module changed since submission -- the redone
  tasks would fill differently from the finished ones. `--allow-config-change`
  overrides that for edits that cannot change the fills (labels, comments).

Exit code: 0 when everything is done, queued or resubmitted; 1 otherwise. Run
it now and then, not in a loop: every call queries the shared scheduler.

### Plotting a run

```bash
suep-plot $OUT/configs_mds                     # config set from run_info.json,
                                               # figures to $OUT/configs_mds/plots
suep-plot $OUT/configs_mds --formats png -j 4  # faster while iterating
```

- `suep-plot` sums the pickles of the run -- the parts of a sharded sample
  become one sample -- and only those this run's tasks wrote: missing, stale or
  foreign pickles are reported, never silently summed. A directory holding
  both `X.pkl` and `X.part<k>.pkl` is refused (that would count events twice).
- Styling (`label`, colours, `rebin`, `log_*`, ...) is re-read from the config
  set each time, so it can change without reprocessing; `-c` picks another set.
- The figures of `<region>/<cut>/<variable>` histograms go into matching
  folders. `index.html` shows them grouped by folder, with a filter box (type
  `iso_muon size`), and only the figures this run wrote.
- `cutflow.txt` / `cutflow.csv`: events passing each selection, per sample.
- `--save-root merged.root` also writes every histogram as `<histogram>/<sample>`.
- `--lumi` scales MC by `xs × lumi × 1000 / sumw`. The MDS sets carry
  placeholder `xs: 1.0` (`suep-plot` warns), so never pass it to them.
- Exit code 1 if no pickle was found or any figure failed.

`bash $OUT/configs_mds/slurm/merge_and_plot.sh` does the same as
`suep-plot $OUT/configs_mds` and forwards extra options.

### Comparing runs

```bash
suep-plot $OUT/run_a $OUT/run_b -o $OUT/cmp -c configs/configs_mds   # overlay samples
python scripts/compare_eps.py $OUT/configs_mds $OUT/configs_mds_minpts50 $OUT/cmp_minpts \
    "min 10" "min 50"                                   # the same histograms, two runs
python scripts/compare_sig_bkg.py $OUT/configs_mds_signal $OUT/configs_mds_data \
    $OUT/sigbkg --sig-cut matched --bkg-cut all          # matched signal vs all background
```

`compare_eps.py` draws each histogram of each sample from both runs,
unit-normalized with their ratio, ranked by how much the shape changed.
`compare_sig_bkg.py` pairs `<region>/<sig-cut>/<variable>` of the signal run
with `<region>/<bkg-cut>/<variable>` of one or more background runs, ranks the
variables by separation and writes `separation.txt`.

---

## Where outputs go

`-o` defaults to `./output` (`suep-plot`: `<run>/plots`). Keep real runs out
of the repo, on group storage -- `/groups` is backed up, `$SCRATCHDIR` is
deleted after 28 days -- one directory per (config set × clustering), e.g.

```
/groups/<group>/<user>/suep_output/<config>_<gen>_<clustering>/   # pickles + slurm/
/groups/<group>/<user>/suep_plots/<study>_<gen>_<clustering>/     # figures to share
```

A run directory knows what made it (`run_info.json`, `slurm/`), so it is the
thing to keep; figures can always be redrawn from it.

---

## Checking a change to the framework

```bash
srun --partition=c --qos=short --time=00:30:00 --mem=8000 --cpus-per-task=4 \
     bash -c 'eval "$(conda shell.bash hook)" && conda activate rootmds && python -m pytest -q'
bash scripts/smoke_test.sh                     # one file, run -> pickle -> plot, ~1 min
python scripts/dump_config.py configs/*/ -o <dir>     # expanded configs; diff before/after
```

- The tests need no input files; run them on a compute node (the login node is
  an order of magnitude slower).
- `smoke_test.sh` writes into a fresh directory under `$SCRATCHDIR` (or the one
  you name, if it is new or empty) and never deletes anything.
- An empty `dump_config.py` diff proves a config-layer change kept every
  expanded definition. For a change to the filling, run one file before and
  after (`suep-run ... --file-range 0:1 --part 0`) and compare every
  histogram's values and variances.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ERROR: no grid proxy` / `the grid proxy ... expires` | Renew it (`voms-proxy-init`, above). `xrootd ... [3010] Operation not permitted` inside a job is the same thing. |
| `ERROR: ... is not a config set` | `-c` names one set, e.g. `configs/configs_mds`; the message lists them. |
| `ERROR: ... is a base set` | `common`, `common_gen`, `common_clustersrc` are only extended; pick a set that extends them. |
| `ERROR in <file>: ...` before anything runs | A config mistake, named with its file -- often with "did you mean". `suep-config check` finds them all at once. |
| `ERROR: config expressions that do not work` | An expression fails on some sample's first events (a typo, or a branch that sample lacks); the list names histogram, sample and expression. |
| `ERROR: unknown sample(s)` | `-s` takes the names in `samples.yaml`; the message lists them. |
| `already holds a run` | Submit into a new `-o`, or `--overwrite`. |
| `array tasks, but Slurm here accepts indices 0-3000` | Use the larger `--files-per-job` the message suggests. |
| `QOSMaxWallDurationPerJobLimit` | Not raised any more: the QOS is picked from `--time`. With an explicit `--qos`, the message says which limit it exceeds. |
| Tasks killed for memory | Raise `--mem` on resubmit. DBSCAN memory grows with the rechits of one event, not with `--chunk-size` (see known-issues.md). |
| `suep-run` says "up to date" | Only for local inputs: pass `--force` after changing code in `src/`. |
| `fork: Resource temporarily unavailable` | Too much running on the login node -- submit instead. |
| Plots have no labels / colours | Plotting without `-c` and without a `run_info.json` (old runs): pass `-c <config set>`. |
| `both a whole-sample pickle and part pickles` | A directory mixing two runs: move the one that does not belong out. |
