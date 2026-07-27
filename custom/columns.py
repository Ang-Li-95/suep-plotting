"""User-defined derived columns (coffea NanoEvents).

The processor calls ``derive(events)`` once per chunk, before filling
histograms, and uses the returned array in place of ``events``.  Add new fields
with :func:`awkward.with_field` and reference them in ``histograms.yaml`` /
``selections.yaml`` expressions as ``events.<name>``.

Return the (possibly augmented) ``events`` array.  If you don't need any derived
columns, the default pass-through below is fine.

MDS LLP cluster study (configs_mds/, configs_mds_shape/)
---------------------------------------------------------
For MDSNANO samples (detected by the presence of the ``cscRechits`` collection)
this module attaches four derived collections:

``events.llp``
    One entry per generated LLP (``SUEPGenPart.pdgId == 999999``): kinematics,
    decay vertex (from the production vertex of its daughters), the 3D
    ``openingAngle`` between the two decay daughters (boost-driven: a more
    boosted LLP gives a tighter pair), decay-volume flags
    (``inCSC/inDT/inRPC``), per-system truth rechit counts
    (``nHitsCSC/nHitsDT/nHitsRPC/nHitsTotal``) and reconstruction flags
    (``recoCSC/recoDT/recoRPC/reco``: LLP has a matched DBSCAN cluster).
    Background samples without ``SUEPGenPart`` get an *empty* llp collection
    so llp-based expressions still evaluate (to nothing).

    Gen-level shape of the matched rechits (suffix ``CSC``/``DT``/``RPC`` for
    one system, ``Total`` for the three pooled): ``drPairMin``/``drPairMax``
    (extreme dR between two of the LLP's rechits) and
    ``dr50``/``dr80``/``dr90``/``drMax`` (radius around the rechit centroid
    holding that percentage of them).  The rechit collections gain
    ``matchedLLP`` (rechit carries an LLP's ``llpIdx``) and ``drLLP`` (its dR
    to that LLP's rechit centroid within the same system).

``events.cscCluster`` / ``events.dtCluster`` / ``events.rpcCluster``
    DBSCAN clusters of the muon-system rechits (per system, dR metric in
    eta-phi with proper phi wrap-around).  A cluster is truth-matched to an
    LLP when at least ``MATCH_MIN_HITS`` of its rechits carry that LLP's
    ``llpIdx``.  On samples without the truth branches (central background
    MDSNano, e.g. DY) clustering still runs; ``matched`` is always False and
    ``hasTruth`` is False, so ``matched | ~hasTruth`` selects "signal-like"
    clusters uniformly (gen-matched in signal, all clusters in background).

    Shape variables (per cluster): hit-count moments across detector layers
    (``nLayer``, ``layerHitsMean/RMS/RelRMS``, ``maxLayerFrac``) and across
    stations (``nStation``, ``stationSpan``, ``avgStation``,
    ``maxStationFrac``), plus spatial spreads (``etaSpread``, ``phiSpread``,
    ``rSpread``, ``zSpread``).  Isolation from prompt activity: ``drMuon`` /
    ``drJet``, the dR from the cluster centroid to the closest reconstructed
    muon / jet (NaN when the event has none, so those clusters simply drop out
    of the histogram).  Two "layer" definitions are kept in parallel:
    ``layer*`` fields use the stored segmentation (physical chamber for CSC,
    which has no in-chamber layer branch; the physical layer for DT/RPC),
    while ``zLayer*`` fields identify the layer plane from quantized global z
    (full in-chamber resolution for CSC, but only on ntuples that store the
    true per-layer z — see ``_flat_zlayer_key``).

Note: ``SUEPGenPart`` is the *full unpruned* genParticles collection; the
rechit ``llpIdx`` branches index into it.  NanoAODSchema attaches no physics
behaviours to it (unknown collection), so energy and mother/daughter relations
are computed by hand.
"""

from __future__ import annotations

import os

import awkward as ak
import numpy as np
from numba import njit

# ── MDS study parameters ───────────────────────────────────────────
LLP_PDGID = 999999

# Decay-volume definitions [cm] (standard MDS analysis geometry)
CSC_ZMIN, CSC_ZMAX, CSC_RMAX = 400.0, 1100.0, 695.5
DT_RMIN, DT_RMAX, DT_ZMAX = 380.0, 738.0, 650.0
RPC_EC_ZMIN, RPC_EC_ZMAX, RPC_EC_RMAX = 600.0, 1020.0, 660.0  # endcap; barrel = DT volume

# DBSCAN (eps in dR, min_samples) per system.  The CSC/DT minimum cluster
# size can be overridden with $MDS_CLUSTER_MIN_SAMPLES (default 50) and the
# dR radius with $MDS_CLUSTER_EPS (default 0.2, all three systems), to study
# their effect without editing the config; RPC min_samples stays at 10
# (sparse system).
_CSC_DT_MIN_SAMPLES = int(os.environ.get("MDS_CLUSTER_MIN_SAMPLES", "50"))
_EPS = float(os.environ.get("MDS_CLUSTER_EPS", "0.2"))
DBSCAN_PARAMS = {
    "csc": (_EPS, _CSC_DT_MIN_SAMPLES),
    "dt": (_EPS, _CSC_DT_MIN_SAMPLES),
    "rpc": (_EPS, 10),
}

# $MDS_SKIP_CLUSTERING=1 drops the DBSCAN step, ~35% of derive() (measured on
# the 2024 signal: 8.8 s -> 5.7 s per 1000 events).
# For gen-level-only configs (configs_mds_gen/): the cluster collections and
# the llp.reco* flags are then NOT attached, so a config that needs them fails
# at expression validation instead of quietly filling empty histograms.
SKIP_CLUSTERING = os.environ.get("MDS_SKIP_CLUSTERING", "0") not in ("0", "", "false")

# Cluster <-> LLP matching: matched := (# rechits sharing one llpIdx) >= MATCH_MIN_HITS
MATCH_MIN_HITS = 10

# Per-LLP rechit spread: an LLP with more than this many matched rechits is
# strided down before the O(N^2) pairwise dR matrix is built (hit
# multiplicities are far below the cap in practice, so this never fires).
PAIR_MAX_HITS = 2000

# Containment fractions for the per-LLP cone radius (field names dr50/dr80/...)
DR_QUANTILES = (0.5, 0.8, 0.9)

# Rechit ``llpIdx`` convention: "genpart" (default; llpIdx is the SUEPGenPart
# index, post-Geant4-fix files) or "ordinal" (llpIdx is the ordinal LLP index
# within the event, pre-fix files).  Override with $MDS_LLPIDX_CONVENTION.
LLPIDX_CONVENTION = os.environ.get("MDS_LLPIDX_CONVENTION", "genpart")
if LLPIDX_CONVENTION not in ("genpart", "ordinal"):
    raise ValueError(
        f"MDS_LLPIDX_CONVENTION must be 'genpart' or 'ordinal', got '{LLPIDX_CONVENTION}'")


def _llpidx_is_genpart_index(events):
    """Detect whether rechit ``llpIdx`` indexes SUEPGenPart directly.

    Post-Geant4-fix files store the SUEPGenPart index (``SUEPGenPart.pdgId`` at
    that index is the LLP, 999999); pre-fix files store the ordinal LLP index
    (which points at low-index beam/hard-process particles instead).  Both
    conventions keep ``llpIdx < nSUEPGenPart``, so indexing is always safe.
    Returns True (genpart-index convention) when no matched hits are present.

    NOT called at runtime: the convention is taken from
    ``$MDS_LLPIDX_CONVENTION`` (default "genpart").  Kept as a standalone
    check for validating that setting on a new production.
    """
    idx = events.cscRechits.llpIdx
    sel = idx[idx >= 0]
    matched_pdg = ak.flatten(events.SUEPGenPart.pdgId[sel])
    if len(matched_pdg) == 0:
        return True
    return bool(ak.mean(matched_pdg == LLP_PDGID) > 0.5)


def derive(events):
    """Attach the MDS LLP/cluster collections (MDSNANO samples only)."""
    if "cscRechits" not in events.fields:
        return events
    has_truth = "SUEPGenPart" in events.fields

    clusters = {}
    if not SKIP_CLUSTERING:
        for sys, coll, timefield in (("csc", "cscRechits", "Tpeak"),
                                     ("dt", "dtRecHits", None),
                                     ("rpc", "rpcRecHits", "Time")):
            eps, min_samples = DBSCAN_PARAMS[sys]
            clusters[sys] = _cluster_system(events[coll], timefield, eps, min_samples)
            for field, obj in (("drMuon", "Muon"), ("drJet", "Jet")):
                if obj in events.fields:
                    clusters[sys] = ak.with_field(
                        clusters[sys], _dr_to_nearest(clusters[sys], events[obj]), field)

    if has_truth:
        llp = _build_llps(events)

        # The rechit truth branch ``llpIdx`` uses one of two conventions
        # depending on the producer: the SUEPGenPart index of the LLP
        # (post-Geant4-fix, SUEPs_Gen2; the default) or the ordinal LLP index
        # within the event (pre-fix, SUEPs_Gen; set
        # MDS_LLPIDX_CONVENTION=ordinal).  Match against the LLP field in the
        # same space.
        match_key = llp.gidx if LLPIDX_CONVENTION == "genpart" else llp.lidx

        # Per-LLP truth rechit counts and reconstruction flags
        nhits = {}
        for sys, coll in (("CSC", "cscRechits"), ("DT", "dtRecHits"), ("RPC", "rpcRecHits")):
            nhits[sys] = ak.values_astype(
                ak.sum(match_key[:, :, None] == events[coll].llpIdx[:, None, :], axis=2),
                np.int64)
        llp = ak.with_field(llp, nhits["CSC"], "nHitsCSC")
        llp = ak.with_field(llp, nhits["DT"], "nHitsDT")
        llp = ak.with_field(llp, nhits["RPC"], "nHitsRPC")
        llp = ak.with_field(llp, nhits["CSC"] + nhits["DT"] + nhits["RPC"], "nHitsTotal")

        if not SKIP_CLUSTERING:
            reco = {}
            for sys, SYS in (("csc", "CSC"), ("dt", "DT"), ("rpc", "RPC")):
                # (ev, nLLP, nCluster) truth-match table: cluster's best-LLP
                # index equals this LLP's.  Unmatched clusters carry -1, which
                # never equals an LLP index (>= 0), so they drop out.
                sel = match_key[:, :, None] == clusters[sys].matchedLLPIdx[:, None, :]
                reco[sys] = ak.any(sel, axis=2)
                # (2) number of DBSCAN clusters truth-matched to each LLP
                nclu = ak.values_astype(ak.sum(sel, axis=2), np.int64)
                llp = ak.with_field(llp, nclu, "nRecoCluster" + SYS)
                # (3) fraction of the LLP's matched rechits captured by its best
                # matched cluster: max over matched clusters of that cluster's
                # matched-hit count, divided by the LLP's total matched rechits
                # in the system.  NaN when the LLP has no matched rechit there.
                hits_in = ak.where(sel, clusters[sys].nMatchedHits[:, None, :], 0)
                best = ak.fill_none(ak.max(hits_in, axis=2), 0)
                nh = llp["nHits" + SYS]
                frac = ak.where(nh > 0, best / ak.where(nh > 0, nh, 1), np.nan)
                llp = ak.with_field(llp, frac, "clusterHitFrac" + SYS)
            llp = ak.with_field(llp, reco["csc"], "recoCSC")
            llp = ak.with_field(llp, reco["dt"], "recoDT")
            llp = ak.with_field(llp, reco["rpc"], "recoRPC")
            llp = ak.with_field(llp, reco["csc"] | reco["dt"] | reco["rpc"], "reco")

    else:
        llp = _empty_llps(events)

    # Eta-phi spread of each LLP's truth-matched rechits, per system and for
    # the three pooled ("Total", matching nHitsTotal).  The per-system passes
    # also write each matched rechit's dR to its LLP's centroid back onto the
    # rechit collection.  Samples without truth branches get the same fields
    # (all NaN / False), so the configs evaluate on background as well.
    nllp = ak.num(llp.pt)
    for sys, coll in (("CSC", "cscRechits"), ("DT", "dtRecHits"),
                      ("RPC", "rpcRecHits")):
        rechits = events[coll]
        counts = ak.num(rechits.Eta)
        hit_dr = np.full(int(ak.sum(counts)), np.nan)
        if has_truth:
            for f, v in _llp_rechit_dr(events, [coll], match_key,
                                       hit_dr=hit_dr).items():
                llp = ak.with_field(llp, ak.unflatten(v, nllp), f + sys)
        matched = (rechits.llpIdx >= 0 if "llpIdx" in rechits.fields
                   else ak.values_astype(ak.zeros_like(rechits.Eta), np.bool_))
        rechits = ak.with_field(rechits, ak.unflatten(hit_dr, counts), "drLLP")
        events = ak.with_field(events, ak.with_field(rechits, matched, "matchedLLP"),
                               coll)
    if has_truth:
        for f, v in _llp_rechit_dr(
                events, ["cscRechits", "dtRecHits", "rpcRecHits"], match_key).items():
            llp = ak.with_field(llp, ak.unflatten(v, nllp), f + "Total")

    events = ak.with_field(events, llp, "llp")
    if SKIP_CLUSTERING:
        # Deliberately no events.<sys>Cluster and no llp.reco* fields: a config
        # that needs them should fail loudly at expression validation rather
        # than silently fill zeros.
        return events
    events = ak.with_field(events, clusters["csc"], "cscCluster")
    events = ak.with_field(events, clusters["dt"], "dtCluster")
    events = ak.with_field(events, clusters["rpc"], "rpcCluster")
    return events


def _dr_to_nearest(clusters, objects):
    """dR from each cluster centroid to the closest object in the event.

    NaN in events with no such object (``ak.min`` over an empty list), which the
    fill drops.  Written out by hand because the cluster collection is a plain
    record array with no vector behaviour, so ``.nearest()`` is unavailable.
    """
    deta = clusters.eta[:, :, None] - objects.eta[:, None, :]
    dphi = (clusters.phi[:, :, None] - objects.phi[:, None, :] + np.pi) % (2 * np.pi) - np.pi
    return ak.fill_none(ak.min(np.sqrt(deta ** 2 + dphi ** 2), axis=2), np.nan)


def _build_llps(events):
    """LLP collection from the unpruned SUEPGenPart table.

    The decay vertex is the production vertex of the LLP's daughters
    (every LLP in these samples has exactly two daughters stored).
    """
    gp = events.SUEPGenPart
    gp_idx = ak.local_index(gp.pdgId, axis=1)
    is_llp = gp.pdgId == LLP_PDGID
    gidx = gp_idx[is_llp]              # SUEPGenPart index of each LLP
    lidx = ak.local_index(gidx, axis=1)  # ordinal 0..nLLP-1 within the event

    mom = gp.genPartIdxMother
    has_mom = mom >= 0
    mom_pdg = gp.pdgId[ak.where(has_mom, mom, 0)]
    is_dau = has_mom & (mom_pdg == LLP_PDGID)
    dau_mom = mom[is_dau]

    # first daughter per LLP: (nLLP x nDau) comparison of int indices only
    match = gidx[:, :, None] == dau_mom[:, None, :]
    has_dau = ak.any(match, axis=2)
    first = ak.fill_none(ak.argmax(match, axis=2), 0)
    dvx = ak.where(has_dau, gp.vx[is_dau][first], np.nan)
    dvy = ak.where(has_dau, gp.vy[is_dau][first], np.nan)
    dvz = ak.where(has_dau, gp.vz[is_dau][first], np.nan)

    # Opening angle between the two decay daughters (every LLP has exactly two).
    # Boost-driven: a more boosted LLP decays into a tighter pair, which is what
    # ultimately sets the eta-phi spread of its rechits.  Second daughter = the
    # matched one that is not `first`; NaN if the LLP has fewer than two stored.
    dau_local = ak.local_index(dau_mom, axis=1)
    match2 = match & (dau_local[:, None, :] != first[:, :, None])
    has_two = ak.sum(match, axis=2) >= 2
    second = ak.fill_none(ak.argmax(match2, axis=2), 0)
    d_eta, d_phi = gp.eta[is_dau], gp.phi[is_dau]
    t1 = 2 * np.arctan(np.exp(-d_eta[first]))
    t2 = 2 * np.arctan(np.exp(-d_eta[second]))
    dphi_d = d_phi[first] - d_phi[second]
    cos_open = (np.sin(t1) * np.sin(t2) * np.cos(dphi_d)
                + np.cos(t1) * np.cos(t2))
    # np.clip is not a ufunc, so it can't dispatch over the jagged array; clamp
    # with the min/max ufuncs before arccos (guards float round-off past +-1).
    cos_open = np.minimum(np.maximum(cos_open, -1.0), 1.0)
    opening_angle = ak.where(has_two, np.arccos(cos_open), np.nan)

    pt, eta, mass = gp.pt[is_llp], gp.eta[is_llp], gp.mass[is_llp]
    decay_r = np.hypot(dvx, dvy)
    abs_z = abs(dvz)

    in_csc = (abs_z > CSC_ZMIN) & (abs_z < CSC_ZMAX) & (decay_r < CSC_RMAX)
    in_dt = (decay_r > DT_RMIN) & (decay_r < DT_RMAX) & (abs_z < DT_ZMAX)
    in_rpc = in_dt | ((abs_z > RPC_EC_ZMIN) & (abs_z < RPC_EC_ZMAX)
                      & (decay_r < RPC_EC_RMAX))

    # Proper decay length: L_proper = L_lab / (beta*gamma) = L_lab * m / p,
    # where L_lab is the 3D distance from the LLP production vertex to its
    # decay vertex.  Distributed exponentially with mean = nominal ctau.
    p = pt * np.cosh(eta)
    l_lab = np.sqrt((dvx - gp.vx[is_llp]) ** 2 + (dvy - gp.vy[is_llp]) ** 2
                    + (dvz - gp.vz[is_llp]) ** 2)
    betagamma = p / mass

    return ak.zip({
        "pt": pt,
        "eta": eta,
        "phi": gp.phi[is_llp],
        "mass": mass,
        "energy": np.hypot(p, mass),
        "gidx": gidx,
        "lidx": lidx,
        "decayR": decay_r,
        "decayZ": dvz,
        "Llab": l_lab,
        "betagamma": betagamma,
        "ctau": l_lab / betagamma,
        "openingAngle": opening_angle,
        "inCSC": in_csc,
        "inDT": in_dt,
        "inRPC": in_rpc,
    })


def _empty_llps(events):
    """Zero-length llp collection (background samples without SUEPGenPart).

    Same fields as the real collection, so llp expressions in the configs
    evaluate to empty results instead of raising on background samples.
    """
    counts = np.zeros(len(events), dtype=np.int64)

    def empty(dtype):
        return ak.unflatten(np.zeros(0, dtype=dtype), counts)

    f64 = (("pt", "eta", "phi", "mass", "energy", "decayR", "decayZ",
            "Llab", "betagamma", "ctau", "openingAngle",
            "clusterHitFracCSC", "clusterHitFracDT", "clusterHitFracRPC")
           + tuple(f + sys for sys in ("CSC", "DT", "RPC", "Total")
                   for f in _dr_field_names()))
    i64 = ("gidx", "lidx", "nHitsCSC", "nHitsDT", "nHitsRPC", "nHitsTotal",
           "nRecoClusterCSC", "nRecoClusterDT", "nRecoClusterRPC")
    boo = ("inCSC", "inDT", "inRPC", "recoCSC", "recoDT", "recoRPC", "reco")
    fields = {f: empty(np.float64) for f in f64}
    fields.update({f: empty(np.int64) for f in i64})
    fields.update({f: empty(np.bool_) for f in boo})
    return ak.zip(fields)


def _dr_field_names():
    """Names of the per-LLP rechit-spread fields (without the system suffix)."""
    return ("drPairMin", "drPairMax", "drMax") + tuple(
        f"dr{int(round(q * 100))}" for q in DR_QUANTILES)


def _llp_rechit_dr(events, coll_names, keys, hit_dr=None):
    """Eta-phi spread of the rechits truth-matched to each LLP.

    Pools the rechit collections named in ``coll_names`` (one system, or all
    three for the combined shape) and, for every LLP with at least one matched
    rechit, computes

    ``drPairMin`` / ``drPairMax``
        smallest / largest dR between any two of the LLP's rechits (closest
        pair and shower "diameter").
    ``dr50`` / ``dr80`` / ``dr90`` / ``drMax``
        radius around the rechit centroid containing 50 / 80 / 90 / 100 % of
        them, i.e. the cone size needed to collect that fraction of the hits.

    LLPs with no matched rechit get NaN in every field (silently dropped at
    fill time); ``drPairMin/Max`` are NaN as well for a single-hit LLP.

    ``keys`` is the per-event list of LLP identifiers the rechit ``llpIdx``
    branches are compared against (``llp.gidx`` or ``llp.lidx``).  ``hit_dr``,
    if given, is a flat per-rechit array (single collection only) filled in
    place with each matched rechit's dR to its LLP's centroid.

    Returns a dict of flat per-LLP arrays, to be unflattened with ``ak.num(keys)``.
    """
    nllp = np.asarray(ak.num(keys))
    keys_flat = np.asarray(ak.flatten(keys), dtype=np.int64)
    llp_off = np.concatenate([[0], np.cumsum(nllp)]).astype(np.int64)

    n_llp = len(keys_flat)
    pair_min = np.full(n_llp, np.nan)
    pair_max = np.full(n_llp, np.nan)
    dr_max = np.full(n_llp, np.nan)
    dr_q = np.full((len(DR_QUANTILES), n_llp), np.nan)

    def result():
        out = {"drPairMin": pair_min, "drPairMax": pair_max, "drMax": dr_max}
        for t, q in enumerate(DR_QUANTILES):
            out[f"dr{int(round(q * 100))}"] = dr_q[t]
        return out

    # Materialize the matched rechits of every requested collection to flat
    # numpy once (per-event awkward indexing is slow), as in _cluster_system.
    evts, etas, phis, idxs, srcs = [], [], [], [], []
    for name in coll_names:
        rechits = events[name]
        if "llpIdx" not in rechits.fields:
            continue
        counts = np.asarray(ak.num(rechits.Eta))
        llpidx = np.asarray(ak.flatten(rechits.llpIdx), dtype=np.int64)
        m = llpidx >= 0
        evts.append(np.repeat(np.arange(len(counts), dtype=np.int64), counts)[m])
        etas.append(np.asarray(ak.flatten(rechits.Eta), dtype=np.float64)[m])
        phis.append(np.asarray(ak.flatten(rechits.Phi), dtype=np.float64)[m])
        idxs.append(llpidx[m])
        srcs.append(np.flatnonzero(m).astype(np.int64))
    if not evts or sum(len(e) for e in evts) == 0:
        return result()

    # One stable sort by event groups the pooled hits the way the per-event
    # loop used to: all CSC hits of the event, then DT, then RPC.
    evt = np.concatenate(evts)
    grp = np.argsort(evt, kind="stable")
    hit_eta = np.concatenate(etas)[grp]
    hit_phi = np.concatenate(phis)[grp]
    hit_idx = np.concatenate(idxs)[grp]
    hit_src = (np.concatenate(srcs)[grp] if hit_dr is not None
               else np.empty(0, dtype=np.int64))
    ev_off = np.searchsorted(evt[grp], np.arange(len(nllp) + 1)).astype(np.int64)

    _dr_kernel(ev_off, hit_eta, hit_phi, hit_idx, hit_src, keys_flat, llp_off,
               np.asarray(DR_QUANTILES, dtype=np.float64),
               pair_min, pair_max, dr_max, dr_q,
               hit_dr if hit_dr is not None else np.empty(0),
               hit_dr is not None)
    return result()


@njit(cache=True)
def _dr_kernel(ev_off, hit_eta, hit_phi, hit_idx, hit_src, keys_flat, llp_off,
               quantiles, pair_min, pair_max, dr_max, dr_q, hit_dr, do_hits):
    """Compiled inner loop of :func:`_llp_rechit_dr` (see it for definitions).

    Every array is flat and preallocated by the caller; the five output arrays
    are written in place.  ``ev_off`` slices the matched-hit arrays per event,
    ``llp_off`` slices ``keys_flat`` and the output rows per event.

    Compiled because the groups are tiny (~3 rechits per LLP): in numpy this
    loop was ~17x slower, spending almost all of it on per-call dispatch rather
    than arithmetic.  Results match the numpy version to the last ulp (numpy
    sums pairwise, this sums in a plain loop).
    """
    two_pi = 2.0 * np.pi
    for i in range(len(ev_off) - 1):
        a0, a1 = ev_off[i], ev_off[i + 1]
        k0, k1 = llp_off[i], llp_off[i + 1]
        if a1 <= a0 or k1 <= k0:
            continue
        n = a1 - a0
        # Group this event's hits by llpIdx with one sort, instead of one mask
        # per LLP (events hold up to O(50) LLPs, most without any rechit).
        order = np.argsort(hit_idx[a0:a1], kind="mergesort")

        a = 0
        while a < n:
            key = hit_idx[a0 + order[a]]
            b = a + 1
            while b < n and hit_idx[a0 + order[b]] == key:
                b += 1
            j = -1
            for t in range(k0, k1):
                if keys_flat[t] == key:
                    j = t
                    break
            if j < 0:              # hits from a gen particle that is not an LLP
                a = b
                continue

            m = b - a
            e = np.empty(m, dtype=np.float64)
            p = np.empty(m, dtype=np.float64)
            for t in range(m):
                g = a0 + order[a + t]
                e[t] = hit_eta[g]
                p[t] = hit_phi[g]

            # Centroid: circular mean in phi, so hits either side of +-pi
            # average correctly (same convention as the DBSCAN clusters).
            sin_sum = 0.0
            cos_sum = 0.0
            eta_sum = 0.0
            for t in range(m):
                sin_sum += np.sin(p[t])
                cos_sum += np.cos(p[t])
                eta_sum += e[t]
            cphi = np.arctan2(sin_sum, cos_sum)
            emean = eta_sum / m

            dr = np.empty(m, dtype=np.float64)
            biggest = 0.0
            for t in range(m):
                dphi = (p[t] - cphi + np.pi) % two_pi - np.pi
                dr[t] = np.hypot(e[t] - emean, dphi)
                if dr[t] > biggest:
                    biggest = dr[t]
            dr_max[j] = biggest
            quants = np.quantile(dr, quantiles)
            for t in range(len(quantiles)):
                dr_q[t, j] = quants[t]
            if do_hits:
                for t in range(m):
                    hit_dr[hit_src[a0 + order[a + t]]] = dr[t]

            if m >= 2:
                step = 1 + (m - 1) // PAIR_MAX_HITS
                lo = np.inf
                hi = -np.inf
                for u in range(0, m, step):
                    for v in range(u + step, m, step):
                        dphi = (p[u] - p[v] + np.pi) % two_pi - np.pi
                        d = np.hypot(e[u] - e[v], dphi)
                        if d < lo:
                            lo = d
                        if d > hi:
                            hi = d
                if hi >= 0.0:
                    pair_min[j] = lo
                    pair_max[j] = hi
            a = b


def _flat_layer_key(rechits):
    """Flat int64 id of the finest depth segment ("layer") per system.

    CSC has no in-chamber layer branch in MDSNano, so its layer unit is the
    physical chamber; DT/RPC use the stored physical layer.  See
    :func:`_flat_zlayer_key` for the complementary global-z definition.
    """
    def flat(field):
        return np.asarray(ak.flatten(rechits[field]), dtype=np.int64)

    f = rechits.fields
    if "IChamber" in f:      # CSC: chamber type (+-11..42) x chamber number
        return (flat("Chamber") + 50) * 100 + flat("IChamber")
    if "SuperLayer" in f:    # DT: wheel/sector/station/superlayer/layer
        return ((((flat("Wheel") + 2) * 15 + flat("Sector")) * 5
                 + flat("Station")) * 20
                + (flat("SuperLayer") - 1) * 4 + flat("Layer"))
    # RPC: region/ring/sector/station/layer
    return (((((flat("Region") + 2) * 8 + flat("Ring") + 2) * 20
              + flat("Sector")) * 5 + flat("Station")) * 3 + flat("Layer"))


def _flat_zlayer_key(rechits):
    """Flat int64 id of the layer plane from the global rechit z (1 mm bins).

    CSC layer planes are normal to the beam, so quantized global z identifies
    the in-chamber layer (6 discrete planes per chamber, 2.54 cm spacing,
    2.2 cm in ME1/1; 132 planes in total) — finer than the chamber unit of
    :func:`_flat_layer_key`.  CAVEAT: the central MDSNano v2 background
    production stores ONE z value per chamber (in-chamber layer information
    is dropped), so the ``zLayer*`` cluster fields are only comparable
    between samples whose ntuples keep the true per-layer z (the private
    signal production does; a reprocessed background sample would).  Only
    meaningful for CSC — for barrel systems z runs along the wires, not
    through the layers.
    """
    return np.round(
        np.asarray(ak.flatten(rechits.Z), dtype=np.float64) * 10.0
    ).astype(np.int64)


def _cluster_system(rechits, timefield, eps, min_samples):
    """DBSCAN-cluster one rechit system -> jagged record array of clusters.

    The rechit branches are materialized to flat numpy once per chunk and
    events are processed as offset slices (per-event awkward indexing is slow).
    """
    from sklearn.cluster import DBSCAN

    counts = np.asarray(ak.num(rechits.Eta))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    eta = np.asarray(ak.flatten(rechits.Eta), dtype=np.float64)
    phi = np.asarray(ak.flatten(rechits.Phi), dtype=np.float64)
    xs = np.asarray(ak.flatten(rechits.X), dtype=np.float64)
    ys = np.asarray(ak.flatten(rechits.Y), dtype=np.float64)
    zs = np.asarray(ak.flatten(rechits.Z), dtype=np.float64)
    station = np.asarray(ak.flatten(rechits.Station))
    layerkey = _flat_layer_key(rechits)
    zlayerkey = _flat_zlayer_key(rechits)
    has_truth = "llpIdx" in rechits.fields
    llpidx = (np.asarray(ak.flatten(rechits.llpIdx)) if has_truth
              else np.full(len(eta), -1, dtype=np.int64))
    tvals = (np.asarray(ak.flatten(rechits[timefield]), dtype=np.float64)
             if timefield is not None else None)

    fields = ["size", "eta", "phi", "x", "y", "z", "r", "etaSpread", "phiSpread",
              "rSpread", "zSpread",
              "nStation", "stationSpan", "avgStation", "maxStationFrac",
              "nLayer", "layerHitsMean", "layerHitsRMS", "layerHitsRelRMS",
              "maxLayerFrac",
              "nZLayer", "zLayerHitsMean", "zLayerHitsRMS", "zLayerHitsRelRMS",
              "maxZLayerFrac",
              "nMatchedHits", "matchedLLPIdx", "matched", "nMatchedLLP", "hasTruth"]
    if tvals is not None:
        fields.append("time")
    out = {f: [] for f in fields}
    nclu = np.zeros(len(counts), dtype=np.int64)

    for i in range(len(counts)):
        n = counts[i]
        if n < min_samples:
            continue
        s = slice(offsets[i], offsets[i + 1])
        e, p = eta[s], phi[s]
        dphi = p[:, None] - p[None, :]
        dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
        dmat = np.sqrt((e[:, None] - e[None, :]) ** 2 + dphi ** 2)
        labels = DBSCAN(eps=eps, min_samples=min_samples,
                        metric="precomputed").fit_predict(dmat)
        for k in range(labels.max() + 1):
            m = labels == k
            cphi = np.arctan2(np.sin(p[m]).sum(), np.cos(p[m]).sum())
            dphic = (p[m] - cphi + np.pi) % (2 * np.pi) - np.pi
            idx = llpidx[s][m]
            idx = idx[idx >= 0]
            if len(idx):
                uniq, cnt = np.unique(idx, return_counts=True)
                best, nbest = int(uniq[cnt.argmax()]), int(cnt.max())
                # How many distinct LLPs clear the match threshold in this
                # cluster (a cluster can overlap more than one LLP's shower).
                nmatched_llp = int((cnt >= MATCH_MIN_HITS).sum())
            else:
                best, nbest, nmatched_llp = -1, 0, 0
            matched = nbest >= MATCH_MIN_HITS
            out["size"].append(int(m.sum()))
            out["eta"].append(float(e[m].mean()))
            out["phi"].append(float(cphi))
            out["x"].append(float(xs[s][m].mean()))
            out["y"].append(float(ys[s][m].mean()))
            out["z"].append(float(zs[s][m].mean()))
            out["r"].append(float(np.hypot(xs[s][m].mean(), ys[s][m].mean())))
            out["etaSpread"].append(float(e[m].std()))
            out["phiSpread"].append(float(np.sqrt(np.mean(dphic ** 2))))
            out["rSpread"].append(float(np.hypot(xs[s][m], ys[s][m]).std()))
            out["zSpread"].append(float(zs[s][m].std()))
            # Hit multiplicity across stations: a shower is contained in one
            # station, a punch-through muon crosses several.
            st = np.abs(station[s][m])
            _, scnt = np.unique(st, return_counts=True)
            out["nStation"].append(len(scnt))
            out["stationSpan"].append(int(st.max() - st.min()))
            out["avgStation"].append(float(st.mean()))
            out["maxStationFrac"].append(float(scnt.max() / scnt.sum()))
            # Hit multiplicity across layers: a muon leaves ~1 hit per layer
            # (uniform), a shower a strongly varying number ("fuzziness").
            _, lcnt = np.unique(layerkey[s][m], return_counts=True)
            lmean = float(lcnt.mean())
            out["nLayer"].append(len(lcnt))
            out["layerHitsMean"].append(lmean)
            out["layerHitsRMS"].append(float(lcnt.std()))
            out["layerHitsRelRMS"].append(float(lcnt.std()) / lmean)
            out["maxLayerFrac"].append(float(lcnt.max() / lcnt.sum()))
            # Same moments with the layer plane taken from quantized global z
            # (CSC only in practice; see _flat_zlayer_key for the caveat on
            # ntuples that store one z per chamber).
            _, zcnt = np.unique(zlayerkey[s][m], return_counts=True)
            zmean = float(zcnt.mean())
            out["nZLayer"].append(len(zcnt))
            out["zLayerHitsMean"].append(zmean)
            out["zLayerHitsRMS"].append(float(zcnt.std()))
            out["zLayerHitsRelRMS"].append(float(zcnt.std()) / zmean)
            out["maxZLayerFrac"].append(float(zcnt.max() / zcnt.sum()))
            out["nMatchedHits"].append(nbest)
            out["matchedLLPIdx"].append(best if matched else -1)
            out["matched"].append(matched)
            out["nMatchedLLP"].append(nmatched_llp)
            out["hasTruth"].append(has_truth)
            if tvals is not None:
                out["time"].append(float(tvals[s][m].mean()))
        nclu[i] = labels.max() + 1

    dtypes = {"size": np.int64, "nStation": np.int64, "stationSpan": np.int64,
              "nLayer": np.int64, "nZLayer": np.int64, "nMatchedHits": np.int64,
              "matchedLLPIdx": np.int64, "matched": np.bool_,
              "nMatchedLLP": np.int64, "hasTruth": np.bool_}
    return ak.zip({f: ak.unflatten(np.asarray(v, dtype=dtypes.get(f, np.float64)), nclu)
                   for f, v in out.items()})
