# MDS LLP cluster study — how to reproduce the plots

Analysis of two SUEP MDSNANO signal points (ggH, mMed=125, mDark=2, cτ=5 m,
`temp=1` / `temp=2`) read over xrootd from VBC EOS. The derived collections
(LLPs, DBSCAN CSC/DT/RPC clusters, cluster↔LLP matching) are built in
[`custom/columns.py`](custom/columns.py); the plots are split over two config
sets by whether they need gen information:

| config | contents | output |
| --- | --- | --- |
| [`configs/configs_mds/`](configs/configs_mds/) | reco-only DBSCAN cluster properties and shower shapes, incl. ΔR to the closest muon / jet — fills identically on samples without truth branches | `output_mds*/` |
| [`configs/configs_mds_grid/`](configs/configs_mds_grid/) | everything truth-dependent: the LLP collection, the truth-matched/unmatched cluster splits, the efficiency chain, the matched fractions and the signal-vs-background overlays; runs the full (mDark, T) grid | `output_mds_grid/` |

Both use the same `derive()`, so pick the config set
by which plots you want; `configs/configs_mds_grid/` is the superset of the fills.

Two configurations are produced:

| folder | DBSCAN min cluster size (CSC/DT) | how |
| --- | --- | --- |
| `output_mds/` | 10 | `configs/configs_mds/columns.yaml` |
| `output_mds_minpts50/` | 50 (standard MDS) | copy of that config with `cluster_min_samples: 50` |

RPC clustering is always `min_samples=10` (sparse system). **Never pass `--lumi`** —
`xs=1.0` in `configs/configs_mds/samples.yaml` is a placeholder and lumi scaling would
distort the Clopper–Pearson efficiency intervals.

---

## 0. One-time setup (environment + grid proxy)

```bash
cd /users/ang.li/public/SUEP/suep-plotting
conda activate mds                            # coffea/uproot/sklearn/hist/mplhep
export X509_USER_PROXY=$HOME/private/.proxy   # xrootd auth for root://eos.grid.vbc.ac.at

# refresh the proxy if voms-proxy-info shows it expired (8-day lifetime):
voms-proxy-init -voms cms --valid 192:00 --vomslife 192:0 -out $HOME/private/.proxy
```

---

## 1. Local / interactive

The quickest way is the wrapper script (does both folders, checks the proxy):

```bash
bash reproduce.sh
```

Or run the steps by hand:

```bash
# clustering as configured in configs/configs_mds/columns.yaml  -> output_mds/
suep-run  -c configs/configs_mds -o output_mds --chunk-size 10000 --workers 8
suep-plot output_mds -o output_mds/plots -c configs/configs_mds -j 8

# variant — DBSCAN min cluster size = 50: a copy of the config, one line changed
cp -r configs/configs_mds configs/configs_mds_minpts50
sed -i 's/cluster_min_samples: 10/cluster_min_samples: 50/' configs/configs_mds_minpts50/columns.yaml
suep-run  -c configs/configs_mds_minpts50 -o output_mds_minpts50 --chunk-size 10000 --workers 8
suep-plot output_mds_minpts50 -o output_mds_minpts50/plots -c configs/configs_mds_minpts50 -j 8

# gen-level LLP / matched-rechit plots only -> output_mds_gen/
# (configs/configs_mds_gen/columns.yaml drops the DBSCAN steps; nothing to export)
suep-run  -c configs/configs_mds_gen -o output_mds_gen --chunk-size 10000 --workers 8
suep-plot output_mds_gen -o output_mds_gen/plots -c configs/configs_mds_gen -j 8
```

Each `suep-run` reads all 50 files over xrootd and takes ~8–9 min/sample with 8
workers; `configs/configs_mds_gen`, which runs no DBSCAN, needs ~3.5 min/sample.
Galleries land at `<output>/plots/index.html`.

### Re-style only (no reprocessing)

If you change **plot-time** keys only (labels, `rebin`, `log_*`, `spans`, or
anything in `configs/configs_mds/derived_plots.yaml`), re-run `suep-plot` on the existing
pickles — seconds, no xrootd:

```bash
suep-plot output_mds          -o output_mds/plots          -c configs/configs_mds -j 8
suep-plot output_mds_minpts50 -o output_mds_minpts50/plots -c configs/configs_mds_minpts50 -j 8
```

Reprocess (`suep-run`) only for **fill-time** changes: anything under `custom/`,
`columns.yaml`, or a histogram's `bins`/`edges`/`selections`/`expression`.

---

## 2. Slurm submission

`suep-submit` writes one array task per sample (plus a merge+plot script) under
`<output>/slurm/` and submits it. Slurm's default `--export=ALL` carries your
**submit-shell** environment to the tasks, so `X509_USER_PROXY` must be
exported *before* you submit. The clustering knobs need no exporting: they live
in the config directory's `columns.yaml`, which the tasks read.

```bash
conda activate mds
export X509_USER_PROXY=$HOME/private/.proxy   # ensure the proxy is valid (voms-proxy-info)
```

### Default (min cluster size 50)

```bash
suep-submit -c configs/configs_mds -o output_mds \
    --partition c --time 02:00:00 --mem 16000 --workers 8 --chunk-size 10000

# after all array tasks finish:
bash output_mds/slurm/merge_and_plot.sh -j 8
```

### Checking a run finished, and redoing what didn't

`merge_and_plot.sh` sums whatever pickles it finds, so a run that lost array
tasks merges into a quietly incomplete sample rather than an error. Check
before merging:

```bash
suep-status -o output_mds
```

It reads `slurm/task_list.txt` — the authority on what the run *should* have
produced — and reports per sample how many tasks landed. A task counts as done
only if its pickle can be read back, so a file left truncated by a killed job
is reported as `unreadable` instead of being merged in. Exit code is 0 when
complete, 1 otherwise, so it can gate the merge:

```bash
suep-status -o output_mds && bash output_mds/slurm/merge_and_plot.sh -j 8
```

To redo just the incomplete tasks — same shard boundaries, same input files,
resubmitted as an `sbatch --array` list over the run's own `job.sh`:

```bash
suep-status -o output_mds --resubmit
```

Add `--dry-run` to see the `sbatch` command without running it, or
`--no-verify` to skip reading the pickles back (faster on large runs, but it
will not catch a truncated file). The rerun re-reads the config directory's
`columns.yaml`, so it clusters like its neighbours as long as that file has not
changed since the run — mixing settings inside one output directory is invisible
in the merged result, so `--resubmit` warns when that file is newer than the
submission.

### Variant (min cluster size 50)

Only difference: its own config directory, one line changed.

```bash
cp -r configs/configs_mds configs/configs_mds_minpts50
sed -i 's/cluster_min_samples: 10/cluster_min_samples: 50/' configs/configs_mds_minpts50/columns.yaml
suep-submit -c configs/configs_mds_minpts50 -o output_mds_minpts50 \
    --partition c --time 02:00:00 --mem 16000 --workers 8 --chunk-size 10000

bash output_mds_minpts50/slurm/merge_and_plot.sh -j 8
```

### Splitting samples by files (more nodes in flight)

By default one array task = one whole sample (2 tasks here). Add
`--files-per-job N` to shard each sample's file list into tasks of N files:

```bash
suep-submit -c configs/configs_mds -o output_mds \
    --partition c --time 01:00:00 --mem 16000 --workers 8 --chunk-size 10000 \
    --files-per-job 5          # 25 files/sample -> 5 tasks/sample -> 10 array tasks

bash output_mds/slurm/merge_and_plot.sh -j 8
```

Each shard writes `<sample>.part<k>.pkl`; `suep-plot` (and therefore
`merge_and_plot.sh`) sums the parts back into one sample automatically —
histograms, sumw, nevents and the cutflow all add. **Don't mix** split and
unsplit pickles for the same sample in one output directory (the sample would
be double counted); use a fresh `-o` directory or delete the old pickles when
switching.

Shards are cut from the file list resolved at submission time and written to
`<output>/slurm/filelists/<sample>.part<k>.txt`; tasks read those lists rather
than re-expanding `files:`, so the split stays fixed even if the dataset grows,
and `sbatch --array=<k> job.sh` reruns a failed shard on the same files.

Notes:
- `suep-submit` always reprocesses (`--force` inside the generated job); the
  per-sample pickles are written to `<output>/`, logs to `<output>/slurm/logs/`.
- `--conda-env` defaults to `mds` (matches setup). Add `--max-concurrent N` to cap
  simultaneous array tasks, or `--dry-run` to generate the scripts without submitting
  (then `sbatch output_mds/slurm/job.sh` yourself).
- `merge_and_plot.sh` forwards extra flags (`"$@"`) to `suep-plot`; do **not** add
  `--lumi`.
- Adjust `--partition` to your cluster; each task is one 500k-event sample and needs
  ~16 GB for 8 workers at chunk size 10000.

---

## RPC-merged clustering (one cluster per shower, with a time)

`configs/configs_mds_rpcmerge/` and `configs/configs_mds_rpcmerge_data/` run the
same study with the RPC rechits **clustered into the system they overlap**
instead of on their own: barrel RPC (`Region == 0`) with DT, endcap RPC
(`|Region| == 1`) with CSC.  A shower crossing both detectors is then one
cluster instead of two that have to be paired up afterwards, and the run has
**only `events.cscCluster` and `events.dtCluster`** — there is no
`events.rpcCluster`, because every RPC rechit already belongs to one of those
two.  One line in `columns.yaml` selects it:

```yaml
parameters:
  rpc_merge: true
```

Each merged cluster gains its RPC content (`nRPCHits`, `rpcHitFrac`,
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
the merge: the two detectors are on different clocks, so the RPC estimate is
reported apart rather than averaged in.  The merged-in RPC hits also keep their
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
`configs_mds_grid/` and `configs_mds_data/` with the RPC-cluster blocks pruned,
the "reconstructable" hit counts widened to the merged system
(`nHitsCSC + nHitsRPCEndcap`, `nHitsDT + nHitsRPCBarrel`) and the RPC timing
histograms added, so everything the two studies share stays identical by
construction:

```bash
python scripts/make_rpcmerge_configs.py           # rewrite the two config dirs
python scripts/make_rpcmerge_configs.py --check   # are they current?
```

Running them is the usual pair of commands:

```bash
suep-run  -c configs/configs_mds_rpcmerge -o output_mds_rpcmerge --chunk-size 10000 --workers 8
suep-plot output_mds_rpcmerge -o output_mds_rpcmerge/plots -c configs/configs_mds_rpcmerge -j 8
```

---

## What lives where

- [`configs/configs_mds/samples.yaml`](configs/configs_mds/samples.yaml) — the mDark=2 pair; `configs/configs_mds_grid/samples.yaml` pulls in the full six-point grid instead.
- [`configs/configs_mds/histograms.yaml`](configs/configs_mds/histograms.yaml) — reco-only cluster fills, incl. `<sys>_cluster_dr_muon` / `_dr_jet`.
- [`configs/configs_mds/selections.yaml`](configs/configs_mds/selections.yaml) — reco-only event masks (`has_<sys>_cluster`).
- [`configs/configs_mds_grid/histograms.yaml`](configs/configs_mds_grid/histograms.yaml) — all truth fills + efficiency num/den pairs; LLP η is signed (60 bins, −3..3), not |η|.
- [`configs/configs_mds_grid/selections.yaml`](configs/configs_mds_grid/selections.yaml) — object/event masks (fiducial, ≥10 hits, matched cluster).
- [`configs/configs_mds_grid/derived_plots.yaml`](configs/configs_mds_grid/derived_plots.yaml) — efficiency plots incl. the factorized chain and detector-station bands, matched fractions, sig-vs-bkg overlays.
- [`custom/columns.py`](custom/columns.py) — `derive()`: the step pipeline that attaches the LLP + cluster collections; parameters and optional steps come from each config's `columns.yaml`.  Its helpers sit next to it, one file per topic: [`params.py`](custom/params.py) (constants, defaults, the `columns.yaml` reader), [`clustering.py`](custom/clustering.py) (DBSCAN), [`llp.py`](custom/llp.py) (`events.llp` and the per-LLP rechit spread), [`isolation.py`](custom/isolation.py) (cluster→prompt-object dR).  Everything stays reachable through `custom.columns`.
- [`configs/configs_mds/columns.yaml`](configs/configs_mds/columns.yaml) — clustering parameters of the reference run; [`configs/configs_mds_gen/columns.yaml`](configs/configs_mds_gen/columns.yaml) is the gen-level (no-DBSCAN) step list.
- [`configs/configs_mds_gen/`](configs/configs_mds_gen) — gen-level-only subset (LLPs + matched rechits), → `output_mds_gen/`.
- [`configs/configs_mds_rpcmerge/`](configs/configs_mds_rpcmerge), [`configs/configs_mds_rpcmerge_data/`](configs/configs_mds_rpcmerge_data) — the RPC-merged study (section above), generated by [`scripts/make_rpcmerge_configs.py`](scripts/make_rpcmerge_configs.py) from the two reference sets; do not edit them by hand.

---

## Event display (r–z rechit picture of the clusters)

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

## Gen-level rechit spread of one LLP

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
filling empty histograms. `configs/configs_mds/` + `configs/configs_mds_grid/` keep the full
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
