# The MDS LLP cluster study

What the config sets in this repo were written for: DBSCAN clusters of the
muon-system (CSC / DT / RPC) rechits in SUEP MDSNano samples, compared between
LLP-decay signal, Drell-Yan, ZeroBias data and Cosmics. The derived collections
(LLPs, clusters, cluster ↔ LLP matching) are built in `custom/` -- see
[derived-columns.md](derived-columns.md); the histograms are organized as
regions (`csc`, `dt`, `llp`, ...) -- see [configuration.md](configuration.md).

| config set | contents |
| --- | --- |
| `configs_mds` | reco-only cluster properties and shower shapes, incl. ΔR to the closest muon / jet -- fills identically on samples without truth |
| `configs_mds_minpts50` | the same with the standard MDS DBSCAN minimum cluster size 50 (CSC/DT); `configs_mds` uses 10 |
| `configs_mds_signal` | everything truth-dependent over the (mDark, T) Gen3 grid: the LLP collection, matched / unmatched cluster cuts, the efficiency chain, matched fractions, signal-vs-background overlays |
| `configs_mds_data` | the reco plots with the isolation cuts, on signal, DY and ZeroBias data |

All use the same `derive()`, so pick the set by which plots you want;
`configs_mds_signal` is the superset of the fills. RPC clustering always uses
`min_samples=10` (sparse system). **Never pass `--lumi`**: `xs: 1.0` is a
placeholder in every MDS set, and lumi scaling would distort the
Clopper-Pearson efficiency intervals.

`bash reproduce.sh <output-root>` submits the three main runs.

---

## RPC with the CSC and DT clusters (a time for each shower)

RPC is the only muon subdetector whose MDSNano rechits carry timing at all, so
it is what can date a CSC or DT cluster. The barrel wheels (`Region == 0`) sit
at the DT radii and the endcap disks (`|Region| == 1`) at the CSC z, so each RPC
rechit pairs with one of the two. Two ways to use that, chosen by `rpc_mode` in
`columns.yaml`; both drop `events.rpcCluster` -- every RPC rechit already
belongs with a CSC or a DT cluster, so clustering it again on its own would
double count it:

| `rpc_mode` | config sets | what happens |
| --- | --- | --- |
| `match` | `configs/common/`, so every study | the DBSCAN is the CSC/DT one and the RPC rechits are **associated afterwards** -- nearest cluster centroid within `cluster_eps` -- so they date a cluster without being able to create one |
| `merge` | `configs_mds_rpcmerge`, `configs_mds_rpcmerge_data` | the RPC rechits go **into the DBSCAN** with the system they overlap, so a shower crossing both detectors is one cluster instead of two, and they count towards the per-LLP hit totals |
| `separate` | none (the module default for a set that does not extend `common`) | three independent DBSCAN runs and a standalone `events.rpcCluster` -- the historical layout |

`match` is what every study uses because it leaves the cluster variables and
the signal region exactly those of the standard MDS analysis: RPC contributes
nothing to the density estimate. That matters because RPC rechit times are flat
in BX in data (~19 % at BX 0 in ZeroBias), so merged, RPC noise can push a
background cluster over `min_samples`. Merging in exchange joins a shower that
straddles the two detectors and lets RPC hits bridge a gap in the CSC/DT hit
density. Positional resolution is *not* what separates them: the ΔR spread of
gen-matched rechits about the shower centroid is 1.4× wider for endcap RPC than
CSC (0.053 vs 0.037) and slightly *tighter* for barrel RPC than DT (0.046 vs
0.057) -- both far inside `eps = 0.4`.

Both modes attach the same fields to the same two collections, so one set of
plots reads either and the two compare plot for plot. Each cluster gains its RPC
content (`nRPCHits`, `rpcHitFrac`, `firstSystem`) and the RPC estimate of
**when** it happened:

| field | meaning |
| --- | --- |
| `rpcBx`, `rpcBxMedian`, `rpcBxSpread` | mean / median / RMS bunch crossing (× 25 ns for a time) |
| `rpcOutOfTimeFrac`, `nRPCHitsBx0` | fraction of the cluster's RPC hits outside the in-time BX, and the count inside it |

The BX fields are NaN on a cluster with no RPC hit, which is why their
variables carry the `<sys>_cluster_has_rpc` cut. The CSC `time` (mean `Tpeak`)
is unchanged by either mode: the two detectors are on different clocks, so the
RPC estimate is reported apart rather than averaged in. `rpcHitFrac` is the RPC
share of the cluster's hits in both modes -- `n / (size + n)` under `match`,
since matching leaves those hits out of `size`. `firstSystem` is always the
primary system under `match`, so the `<sys>_cluster_first_not_rpc` cut simply
passes everything there.

Under `merge` the RPC hits inside a cluster keep their own layer and chamber
ids (offset by +1000 in `firstChamber`, so a cluster whose innermost hit is an
RPC one lands in the overflow of the chamber-code plots), so `nLayer`,
`firstChamber` and `firstStation` stay comparable with a `match` run except for
those hits. The first-chamber / first-station variables are cut to clusters
whose innermost hit belongs to the primary system (`<sys>_cluster_first_not_rpc`)
-- on one signal file that is 93 % of the CSC clusters but only 70 % of the DT
ones, since the barrel RPC layers sit inside the DT stations;
`<sys>_cluster_first_system` counts all three.

> **The RPC rechit time is not filled in any MDSNano production, so nothing is
> derived from it.** `rpcRecHits_Time` is exactly 0 with `TimeError` exactly
> −1 throughout -- checked over 841k rechits on the Gen3 signal, on central DY
> and on ZeroBias 2024C -- while `Bx` spans −3..4. The RPC time estimate is
> `rpcBx * 25` ns, the variable `<sys>_cluster_rpc_time_from_bx`. If a
> production ever fills `rpcRecHits_Time`, add the fields back in
> `custom/clustering.py:_rpc_time`;
> `test_no_cluster_field_is_derived_from_the_rpc_rechit_time` is the reminder
> that they are gone on purpose.

The two `rpcmerge` sets extend `configs_mds_signal` / `configs_mds_data` and
change only `rpc_mode` and, for the signal, the "reconstructable" hit counts:
those widen to the merged system (`nHitsCSC + nHitsRPCEndcap`,
`nHitsDT + nHitsRPCBarrel`, the variables `llp_nhits_csc_rpc` /
`llp_nhits_dt_rpc`), so everything the two studies share stays identical by
construction.

---

## Rejecting out-of-time background

Timing here rejects *out-of-time* activity -- previous/later-BX pile-up, cavern
background, noise. In-time pile-up is not a timing problem; that is what the
isolation and shape variables are for.

On **CSC** the handle is the rechit time, and it is the cluster's *spread*, not
its mean, that discriminates (on one signal file: gen-matched clusters have an
RMS of 5.9 ns, unmatched ones 18.3 ns, with both means at ~1.5 ns). Every
cluster of a system whose rechits carry a time therefore gets

| field | meaning |
| --- | --- |
| `time` | mean rechit time (CSC `Tpeak`) |
| `timeSpread` | RMS of those times over the cluster |
| `ootHitFrac` | fraction of them outside ±`oot_time_cut` (12.5 ns by default) |

On **DT** there is no rechit time at all, so the equivalent handle is the RPC
content of the cluster -- `rpcBxSpread`, `rpcOutOfTimeFrac`, `nRPCHitsBx0` --
which is how the standard CMS DT muon-shower search times its clusters. The
same split shows up there: matched clusters have a BX RMS of 0.31 against 0.82
for unmatched ones, while both means sit at ~0.1.

> **Do not take the rejection factor from MC.** The BX composition is wildly
> different in data: BX = 0 holds 77 % of the RPC hits in the signal MC, 61 %
> in DY, but only 19 % in ZeroBias 2024C, where the distribution is essentially
> flat over −2…+2 (an unmodelled uniform cavern/noise background filling the
> readout window). The CSC picture matches: `|Tpeak| < 12.5` ns keeps 61 % of
> signal-MC rechits and 22 % of ZeroBias ones. The real rejection is therefore
> much *larger* than MC suggests -- measure it in ZeroBias, and take only the
> signal efficiency from MC.

---

## Where a cluster comes from: six populations, one set of plots

A DBSCAN cluster in the muon system is not one thing. The search wants to know
what the signal shower looks like *against each of the things that also make
clusters*, separately -- a punch-through jet and a cavern-background blob are
different backgrounds and need not be separated by the same variable. So six
populations are filled into the same histograms and overlaid:

| set | sample | the clusters it fills |
| --- | --- | --- |
| `configs_mds_src_signal` | Gen3 signal, mDark=2, T=1 **and** T=2 | gen-matched: ≥ `match_min_hits` of the cluster's rechits carry one LLP's `llpIdx` |
| `configs_mds_src_dy_jet` | DY→2μ | ΔR(cluster, jet) < 0.4 -- punch-through and hadronic-shower tails |
| `configs_mds_src_dy_muon` | DY→2μ | ΔR(cluster, muon) < 0.4 -- a real muon crossing the chambers, and what it radiates |
| `configs_mds_src_dy_iso` | DY→2μ | ΔR ≥ 0.4 from **both** -- the leftover population as the simulation has it |
| `configs_mds_src_zerobias` | ZeroBias 2024C | ΔR ≥ 0.4 from **both** -- the same cut in minimum-bias data |
| `configs_mds_src_cosmics` | Cosmics 2024C | every cluster, no ΔR requirement -- the non-collision control |

The jet and the muon are the ones `jet_sel_foriso` / `muon_sel_foriso` pick
(the same cuts as everywhere else, from `configs/common/`), and the ΔR is the
`cluster_isolation` step's `drJet` / `drMuon`. 0.4 is the threshold of the
`<sys>_cluster_iso_*` cuts, taken as `< 0.4` here and `≥ 0.4` there, so "in a
jet" and "isolated from jets" partition the clusters with no gap. The in-jet
and on-muon DY categories *overlap* -- a muon inside a jet puts one cluster in
both -- which is the point: they are two sources, not a partition. The
DY-isolated and ZeroBias categories are the same cut on simulation and on data,
so read as a pair they say how much of the leftover population the simulation
reproduces, and how much of the ZeroBias one is the unmodelled cavern and
noise background. Cosmics cannot have a ΔR requirement: that ntuple has no
`Jet` and no `Muon` collection, so its set leaves out the two ΔR variables and
runs `steps: [clusters]`.

**How it is put together.** The population is one named object-level
selection, `csc_cluster_src` / `dt_cluster_src`, which `common_clustersrc` uses
as the `src` cut of the `csc` and `dt` regions and each of the six sets
*defines* in its own `selections.yaml`. So a set is one expression plus one
sample: the binning, the expressions and the clustering parameters are
inherited and identical across all six, which makes the curves comparable bin
for bin. Every histogram is `<sys>/src/<variable>` in all six sets; the
category is carried by the **sample name**, which is why the samples are
aliases named after their population (`clusters_llp`, `clusters_dy_in_jet`,
`clusters_dy_on_muon`, `clusters_dy_iso`, `clusters_zerobias_iso`,
`clusters_cosmics`). The three DY sets are the same dataset under three cuts;
only the distinct config names keep them apart. The base defines *no*
`<sys>_cluster_src` on purpose: a set that forgets stops at startup with
"unknown selection" instead of filling every cluster.

Colours are read at plot time: the five backgrounds take five distinct hues
(DY-in-jet orange, DY-on-muon purple, DY-isolated brown, ZeroBias black,
Cosmics green); the signal points share `tab:blue` and differ by linestyle,
following the registry's convention that colour encodes mDark and linestyle
encodes T.

Each set gets its own run directory, since a run directory holds one run. The
shard sizes differ per sample -- a Cosmics file is ~680k dense rechit events
and one alone outruns the 8 h `short` QOS, a ZeroBias file only ~20k -- and the
QOS follows from `--time`:

```bash
OUT=/groups/<group>/<user>/suep_output/clustersrc_gen3_minpts10_dr04
for s in signal dy_jet dy_muon dy_iso; do
    suep-submit -c configs/configs_mds_src_$s -o $OUT/$s --time 08:00:00 --files-per-job 5
done
suep-submit -c configs/configs_mds_src_zerobias -o $OUT/zerobias --time 08:00:00 --files-per-job 25
suep-submit -c configs/configs_mds_src_cosmics  -o $OUT/cosmics  --time 24:00:00 --mem 16000 \
    --files-per-job 1

for d in signal dy_jet dy_muon dy_iso zerobias cosmics; do suep-status -o $OUT/$d; done

python scripts/compare_sig_bkg.py $OUT/signal \
    $OUT/dy_jet $OUT/dy_muon $OUT/dy_iso $OUT/zerobias $OUT/cosmics $OUT/plots \
    --sig-cut src --bkg-cut src --tag '' --bkg-tag '' \
    -c configs/configs_mds_src_signal configs/configs_mds_src_dy_jet \
       configs/configs_mds_src_dy_muon configs/configs_mds_src_dy_iso \
       configs/configs_mds_src_zerobias configs/configs_mds_src_cosmics
```

`--tag '' --bkg-tag ''` drop the population tag the legend would otherwise
append, since here each *sample* is the population. Every figure is unit-area
normalized, the gallery is ordered by separation, and `separation.txt` scores
the signal against **each** of the five backgrounds -- the number the study is
for. `suep-plot $OUT/signal $OUT/dy_jet ... -o $OUT/plots_raw --normalize`
gives the same curves through the ordinary plotter if the ranking is not
wanted. DY is processed three times, once per category -- the price of the
three populations being three samples.

---

## Event display (r–z rechit picture of the clusters)

`scripts/event_display.py` draws one figure per event with every CSC/DT rechit
at its global (|z|, r), coloured by DBSCAN cluster -- the analysis's own
clustering (`-c <config set>` reads that set's `columns.yaml` and samples). The
marker is the hit-level truth: `o` for a rechit carrying an `llpIdx`, `x` for
one that does not, so the LLP shower and the activity DBSCAN swept up with it
stay visible separately. Noise hits are light grey dots, the grey boxes are the
ME/MB chambers and the solenoid, and an open star marks the truth shower
position (`llpSim*`) of each truth-matched cluster. RPC is not drawn, so a set
with `rpc_mode: merge` is refused. The chamber layout is a z > 0 quarter view,
so the display is folded to |z|.

```bash
python scripts/event_display.py -c configs/configs_mds -d suep_mDark2_temp1 -n 5 \
    --matched-only --min-size 50 --zoom --with-etaphi -o event_display
```

Figures land in `-o` (default `./event_display`) as
`evd_<dataset>_<file>_ev<entry>.png`. `--zoom` crops to the clustered hits (a
shower is a few tens of cm across), `--with-etaphi` adds the η–φ panel the
clustering runs in, `--matched-only` / `--min-size` pick interesting events,
`--systems csc` restricts to one system, and `-f <file.root> --entries 3 7`
draws specific entries of a specific file. It works on background and data too
(`-d dy_2mu_50to120`, `-d zerobias_2024C`); those have no truth branches, so
every cluster is "unmatched".

---

## Gen-level rechit spread of one LLP

`configs_mds_gen` adds, on top of `configs_mds_signal`'s fills, the LLP
shower-shape family: per-LLP matched-rechit counts, the ΔR spread of an LLP's
matched rechits about their centroid, and 2D maps of CSC nHits and ΔR₉₀ against
|η| / pT / energy / boost (their `profile_x` curves are in its
`derived_plots.yaml`). They answer "how wide is the rechit shower of a single
LLP" from truth only (`cscRechits_llpIdx` etc.), with no clustering involved.
Per LLP, over the rechits carrying its `llpIdx`:

| variable | field | meaning |
| --- | --- | --- |
| `llp_nhits_<sys>` | `nHits<SYS>` | matched rechits per LLP |
| `llp_drpair_min/max_<sys>` | `drPairMin/Max<SYS>` | smallest / largest ΔR between two of its rechits |
| `llp_dr{50,80,90}_<sys>`, `llp_drmax_<sys>` | `dr50/dr80/dr90/drMax<SYS>` | radius around the rechit centroid holding 50/80/90/100 % of them |
| `<sys>_rechit_dr_llp` | `<coll>.drLLP` | hit-weighted: every matched rechit's ΔR to its LLP's centroid |

`<sys>` is `csc`, `dt`, `rpc` or `total` (the three pooled); the histograms
are `llp/<cut>/<variable>`, the `hits10_*` cuts keeping only LLPs with ≥ 10
matched rechits -- the number a DBSCAN cluster needs to be truth-matched, and
the regime where a cone size is meaningful. LLPs with < 2 matched rechits have
no ΔR and drop out of the fill.

The maps show that the CSC is an endcap: both the hit count and ΔR₉₀ turn on at
|η| ≈ 1, plateau across the endcap and fall off past |η| ≈ 2.4, in lockstep --
so |η| drives both axes of `llp_dr90_vs_nhits_csc`. At fixed nHits, ΔR₉₀ is
larger at high |η| (the η-φ metric stretches toward the beamline). nHits
saturates with LLP energy (~40 hits) while ΔR₉₀ peaks near 20 GeV and then
declines. The gen `llp.openingAngle` falls as a clean 1/βγ curve, but ΔR₉₀ vs
opening angle is **non-monotonic**: it peaks near 0.3 rad and falls off on both
sides -- toward small angle (high boost) the two daughter showers merge and
ΔR₉₀ drops to the single-shower floor (~0.08 at these |η|), toward wide angle
the rechits are dominated by a single daughter. The controlled
`llp_dr90_vs_betagamma_ctrl` (1.6 < |η| < 2.2, nHits < 25) isolates the boost
side: past the βγ ≈ 5 peak, ΔR₉₀ falls with boost as expected, so a raw
ΔR₉₀-vs-βγ profile is confounded (|η| and nHits both rise with βγ).

Results on `suep_mDark2_temp1` (500k events, median over LLPs with the
[16 %, 84 %] band); `suep_mDark2_temp2` agrees to within ≈ 0.01 everywhere:

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
