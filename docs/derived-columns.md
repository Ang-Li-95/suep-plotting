# Derived columns (`custom/`)

`custom/columns.py` attaches the MDS analysis objects — the generated LLPs and
the DBSCAN clusters of the muon-system rechits — to the NanoEvents array before
histograms are filled. The processor calls it once per chunk:

```python
events = derive(events, settings)
```

`settings` is a `custom.params.Settings`, produced by `configure()` from the
config set's `columns.yaml`. It is immutable and picklable: the processor
resolves it once and pickles it to every worker, so nothing below `derive()`
reads module state and two configurations can run side by side.

Samples without a `cscRechits` collection pass straight through.

---

## `columns.yaml`

Drop it next to `histograms.yaml`. Two keys, both optional:

```yaml
parameters:
  cluster_eps: 0.4
  rpc_mode: merge
steps: [llp, llp_hits, llp_shape]      # gen-level only, no DBSCAN
```

Unknown keys, unknown parameters and unsatisfied step dependencies are **fatal**
— a silently ignored typo here means silently wrong histograms. A config set can
inherit another set's settings with `_extends: ../configs_mds_signal` and state
only what it changes (see [configuration.md](configuration.md)).

### Parameters

The schema is `PARAM_SPEC` in `custom/params.py`; one row per parameter, giving
its default and its validator.

| Parameter | Default | Meaning |
|---|---|---|
| `cluster_eps` | `0.4` | DBSCAN eps — the dR radius in eta-phi, all systems |
| `cluster_min_samples` | `10` | DBSCAN minimum cluster size, CSC and DT (50 = standard MDS analysis) |
| `rpc_min_samples` | `10` | the same for RPC, which is a sparser system; unused outside `rpc_mode: separate` |
| `rpc_mode` | `match` | what the RPC rechits are for — see below |
| `match_min_hits` | `10` | cluster ↔ LLP truth match: `matched` := (# rechits sharing one `llpIdx`) ≥ this |
| `oot_time_cut` | `12.5` | half-width [ns] of the in-time window behind `cluster.ootHitFrac`. 12.5 ns = half a bunch crossing, the standard MDS in-time definition. Only reaches systems with a rechit time, i.e. CSC `Tpeak` |
| `pair_max_hits` | `2000` | an LLP with more matched rechits than this is strided down before the O(N²) pairwise dR matrix is built (multiplicities are far below the cap in practice) |
| `dr_quantiles` | `[0.5, 0.8, 0.9]` | containment fractions for the per-LLP cone radius → the `dr50`/`dr80`/`dr90` field names |
| `llpidx_convention` | `genpart` | how the rechit `llpIdx` branch is read: `genpart` (it is the `SUEPGenPart` index, post-Geant4-fix files) or `ordinal` (the ordinal LLP index within the event, pre-fix files) |
| `jerc` | `true` | apply L1L2L3Res JEC to `events.Jet` before the jets are selected, plus JER smearing on MC |
| `jerc_era` | `2024_Summer24` | jsonpog era for the above |
| `jerc_algo` | `AK4PFPuppi` | jet algorithm for the above |
| `jerc_data_tag` | `null` | overrides the data JEC tag; `null` takes the era entry from `suep_plot.jme.DEFAULTS` |

### `rpc_mode`

RPC is the only muon subdetector whose MDSNano rechits carry timing at all, so
it is what can date a CSC or DT cluster. The barrel wheels (`Region == 0`) sit
at the DT radii and the endcap disks (`|Region| == 1`) at the CSC z, so each RPC
rechit pairs with one of the two.

- **`match`** *(the default)* — the clustering is the CSC/DT one and the RPC
  rechits are associated to the finished clusters afterwards (nearest centroid
  within `cluster_eps`). RPC contributes nothing to the DBSCAN density
  estimate, so the cluster variables and the signal region stay those of the
  standard MDS analysis — which matters because RPC rechit times are flat in
  BX in data (~19 % at BX 0 in ZeroBias), so merged, RPC noise can push a
  background cluster over `min_samples`.
- **`separate`** — the historical layout: three independent DBSCAN runs, and
  `events.rpcCluster` alongside the other two. Only the retired config sets
  still use it; nothing in `configs/common/` defines an `rpc_cluster_*`
  histogram, so a set switching back to it needs its own.
- **`merge`** — the RPC rechits are clustered *together with* the system they
  overlap, so a shower seen by both becomes one cluster. Because they are inside
  the cluster, they also count towards the per-LLP hit totals that define
  "reconstructable in this system".
`merge` and `match` both produce `cscCluster` and `dtCluster` **only** — every
RPC rechit has already been offered to one of those two, and clustering it again
on its own would double count it — and both attach the same fields
(`clustering.RPC_CLUSTER_FIELDS`): the hit content (`nRPCHits`, `nRPCHitsBx0`,
`rpcHitFrac`, `firstSystem`) and the bunch crossing (`rpcBx`, `rpcBxMedian`, `rpcBxSpread`, `rpcOutOfTimeFrac`). So
one config set plots either, and the two can be compared rather than argued
about.

> Nothing is derived from the RPC rechit *time*: no production fills it
> (`Time` exactly 0, `TimeError` exactly -1, over 841k rechits across signal,
> DY and ZeroBias), so the BX is the only real measurement. See
> `clustering._rpc_time`.

### Prompt objects live in `selections.yaml`

Which reconstructed objects count as prompt activity for the cluster isolation
is not a parameter — it is an object selection, so it is written where every
other object selection is written:

```yaml
# configs/configs_mds/selections.yaml
muon_sel_foriso:
  level: object
  collection: Muon
  expression: "(events.Muon.pt > 10) & (abs(events.Muon.eta) < 2.4) & events.Muon.looseId"

jet_sel_foriso:
  level: object
  collection: Jet
  expression: "(events.Jet.pt > 20) & (abs(events.Jet.eta) < 2.4) & (events.Jet.neHEF < 0.8) & (events.Jet.chHEF > 0.1) & events.Jet.tightLepVetoId"
```

The names are fixed — they are the values of `helpers.ISO_SELECTIONS`, which
pairs each with the cluster field it produces (`drMuon`, `drJet`). The
`cluster_isolation` step fails loudly if a config set enabling it does not
define both.

`collection:` says which collection the per-object mask is over, which is what
lets the step apply the mask. The expression is the ordinary config dialect —
`events.Muon.pt`, exactly as in `histograms.yaml`; there is no separate `obj`
alias and no function available only here.

Because these are ordinary selections, they are also usable as ordinary
selections: a histogram may list them under `selections:`, and the cutflow
reports how many events contain such an object.

The step also attaches the **multiplicity** of each prompt collection as an
event-level column named after the selection — `events.n_muon_sel_foriso`,
`events.n_jet_sel_foriso`. That is what the `n_jet_sel_foriso` histogram plots:
the count and the isolation come from the same selection, so retuning the cut
moves both and they cannot disagree. (A collection absent from the file gets no
column, so an expression referencing it fails at validation rather than
plotting zeros.)

Each config set spells the pair out, so a cut can be retuned in one study
without moving the others. `configs_mds`, `configs_mds_signal` and
`configs_mds_data` are overlaid on one canvas, though, so when you retune those
three, retune all three — `test_the_overlaid_sets_share_one_prompt_object_definition`
fails if they drift apart.

> JEC/JER rescale pT and mass but leave eta/phi untouched, so corrections reach
> the cluster dR only through which jets pass the pT threshold above — which is
> exactly what the threshold is for, so the `jerc` step runs first.

### `jet_id`: the ID as a column

2024 NanoAOD no longer stores `Jet_jetId`, so the ID is recomputed from the PF
energy fractions and multiplicities with the official jsonpog `jetid.json.gz`.
The `jet_id` step attaches the decision to `events.Jet` as two plain booleans:

| Field | jsonpog key |
|---|---|
| `events.Jet.tightId` | `AK4PUPPI_Tight` |
| `events.Jet.tightLepVetoId` | `AK4PUPPI_TightLeptonVeto` |

Attaching it as a column rather than exposing a `jet_id()` function to one
privileged expression scope is what lets the jet cut live in `selections.yaml`
at all — and it makes the flags readable from `histograms.yaml` too. The
thresholds stay in the central payload, resolved from `jerc_era` so the ID and
the calibration come from one campaign directory.

Jet ID is JEC-invariant (energy fractions, multiplicities and eta do not move
under a pT rescale), so it does not matter whether `jerc` ran first.

---

## Steps

`steps:` selects a subset. `configs/common/` enables the four reco steps and
`configs/common_gen/` adds the four truth ones, so a study inherits the right
list rather than spelling it out. They always run in this order.
Omitting a step means the fields it attaches are **absent**, so a config that
references them fails loudly at expression validation instead of quietly filling
empty histograms. Dropping `clusters` saves ~35 % of `derive()` (measured on the
2024 signal: 8.8 s → 5.7 s per 1000 events).

| Step | Needs | Attaches |
|---|---|---|
| `jerc` | — | JEC on `events.Jet` in place, plus JER smearing on MC |
| `jet_id` | — | `events.Jet.tightId` / `.tightLepVetoId` from the official payload |
| `clusters` | — | `events.<sys>Cluster` (two systems outside `rpc_mode: separate`) |
| `cluster_isolation` | `clusters` | `drMuon` / `drJet` on the clusters, against the `selections.yaml` prompt objects, plus `events.n_<selection>` per prompt collection |
| `llp` | — | `events.llp`: kinematics, decay vertex, volume flags |
| `llp_hits` | `llp` | `llp.nHits{CSC,DT,RPC,RPCBarrel,RPCEndcap,Total}` |
| `llp_reco` | `clusters`, `llp_hits` | `llp.reco*`, `nRecoCluster*`, `clusterHitFrac*` |
| `llp_shape` | `llp` | per-LLP rechit spread (`dr*`, `drPair*`) and the rechit-level `matchedLLP` / `drLLP` |

---

## The fields

### `events.llp`

One entry per generated LLP (`SUEPGenPart.pdgId == 999999`).

- **Kinematics and vertex** — the decay vertex comes from the production vertex
  of the LLP's daughters; `openingAngle` is the 3D angle between the two decay
  daughters (boost-driven: a more boosted LLP gives a tighter pair).
- **Volume flags** — `inTracker` / `inCalo` / `inCSC` / `inDT`, plus the
  decay point's own `decayEta`.  Each is the *active* volume, not the outer
  envelope: `inTracker` and `inCSC` carry the eta acceptance of the
  subdetector, without which a decay near the beamline counts as fiducial
  where no silicon or chamber reaches it.  There is no `inRPC` — the RPC
  decay volume was the DT volume unioned with an endcap region already
  inside the CSC one, so it added no reach of its own.
- **Truth rechit counts** (`llp_hits`) — `nHitsCSC` / `nHitsDT` / `nHitsRPC` /
  `nHitsRPCBarrel` / `nHitsRPCEndcap` / `nHitsTotal`. Count `llpIdx >= 0` on the
  rechits; never sum these to get an event total, which double counts.
- **Reconstruction flags** (`llp_reco`) — `recoCSC` / `recoDT` / `recoRPC` /
  `reco`: the LLP has a truth-matched DBSCAN cluster. Plus `nRecoCluster<SYS>`
  (how many) and `clusterHitFrac<SYS>` (what fraction of the LLP's matched
  rechits its best matched cluster captured; NaN when it has none there).
- **Shape** (`llp_shape`) — `drPairMin` / `drPairMax` (extreme dR between two of
  the LLP's rechits) and `dr50` / `dr80` / `dr90` / `drMax` (radius around the
  rechit centroid holding that fraction of them). Suffixed `CSC`/`DT`/`RPC` for
  one system, `Total` for the three pooled. The rechit collections themselves
  gain `matchedLLP` (this rechit carries an LLP's `llpIdx`) and `drLLP` (its dR
  to that LLP's rechit centroid within the same system).

Background samples without `SUEPGenPart` get an *empty* `llp` collection with the
same fields, so the same configs evaluate on background.

> `SUEPGenPart` is the *full unpruned* genParticles collection and the rechit
> `llpIdx` branches index into it. NanoAODSchema attaches no physics behaviours
> to it (unknown collection), so energy and mother/daughter relations are
> computed by hand.

### `events.<sys>Cluster`

DBSCAN clusters of the muon-system rechits, per system, dR metric in eta-phi with
proper phi wrap-around.

- **Truth match** — a cluster is matched to an LLP when at least
  `match_min_hits` of its rechits carry that LLP's `llpIdx`. On samples without
  the truth branches (central background MDSNano, e.g. DY) clustering still runs;
  `matched` is always False and `hasTruth` is False, so `matched | ~hasTruth`
  selects "signal-like" clusters uniformly — gen-matched in signal, all clusters
  in background.
- **Where it starts** — `firstChamber` / `firstStation`, the chamber-type code
  and station of the hit closest to the interaction point. A punch-through jet
  starts in an innermost chamber; a genuine displaced shower need not.
- **Timing** — on systems whose rechits carry a time (CSC `Tpeak`): `time`,
  `timeSpread` and `ootHitFrac`, the mean, the RMS and the fraction of hits
  outside ±`oot_time_cut`. The last two are the out-of-time-pile-up handles. DT
  rechits have no time branch at all; there the equivalent is the RPC content of
  a merged cluster (`rpcBxSpread`, `rpcOutOfTimeFrac`, `nRPCHitsBx0`).
- **Shape** — hit-count moments across detector layers (`nLayer`,
  `layerHitsMean` / `RMS` / `RelRMS`, `maxLayerFrac`) and across stations
  (`nStation`, `stationSpan`, `avgStation`, `maxStationFrac`), plus spatial
  spreads (`etaSpread`, `phiSpread`, `rSpread`, `zSpread`).
- **Isolation** — `drMuon` / `drJet`: the dR
  from the cluster centroid to the closest such object, or `NO_OBJECT_DR` (999)
  in events that have none — so an isolation cut is a plain `drMuon >= x` with no
  special case.

Two "layer" definitions are kept in parallel: `layer*` fields use the stored
segmentation (physical chamber for CSC, which has no in-chamber layer branch; the
physical layer for DT/RPC), while `zLayer*` fields identify the layer plane from
quantized global z — full in-chamber resolution for CSC, but only on ntuples that
store the true per-layer z (see `_flat_zlayer_key`).
