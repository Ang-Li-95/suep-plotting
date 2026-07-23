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
    decay vertex (from the production vertex of its daughters), decay-volume
    flags (``inCSC/inDT/inRPC``), per-system truth rechit counts
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
    ``rSpread``, ``zSpread``).  Two "layer" definitions are kept in parallel:
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
    for sys, coll, timefield in (("csc", "cscRechits", "Tpeak"),
                                 ("dt", "dtRecHits", None),
                                 ("rpc", "rpcRecHits", "Time")):
        eps, min_samples = DBSCAN_PARAMS[sys]
        clusters[sys] = _cluster_system(events[coll], timefield, eps, min_samples)

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

        reco = {}
        for sys in ("csc", "dt", "rpc"):
            reco[sys] = ak.any(
                match_key[:, :, None] == clusters[sys].matchedLLPIdx[:, None, :], axis=2)
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
    events = ak.with_field(events, clusters["csc"], "cscCluster")
    events = ak.with_field(events, clusters["dt"], "dtCluster")
    events = ak.with_field(events, clusters["rpc"], "rpcCluster")
    return events


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
            "Llab", "betagamma", "ctau")
           + tuple(f + sys for sys in ("CSC", "DT", "RPC", "Total")
                   for f in _dr_field_names()))
    i64 = ("gidx", "lidx", "nHitsCSC", "nHitsDT", "nHitsRPC", "nHitsTotal")
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
    fields = _dr_field_names()
    nllp = np.asarray(ak.num(keys))
    keys_flat = np.asarray(ak.flatten(keys))
    llp_off = np.concatenate([[0], np.cumsum(nllp)])
    out = {f: np.full(len(keys_flat), np.nan) for f in fields}

    # Materialize the rechit branches to flat numpy once (per-event awkward
    # indexing is slow), as in _cluster_system.
    per_coll = []
    for name in coll_names:
        rechits = events[name]
        if "llpIdx" not in rechits.fields:
            continue
        counts = np.asarray(ak.num(rechits.Eta))
        per_coll.append((
            np.concatenate([[0], np.cumsum(counts)]),
            np.asarray(ak.flatten(rechits.Eta), dtype=np.float64),
            np.asarray(ak.flatten(rechits.Phi), dtype=np.float64),
            np.asarray(ak.flatten(rechits.llpIdx)),
        ))
    if not per_coll:
        return out

    for i in range(len(nllp)):
        if nllp[i] == 0:
            continue
        etas, phis, idxs, srcs = [], [], [], []
        for offsets, eta, phi, llpidx in per_coll:
            s = slice(offsets[i], offsets[i + 1])
            m = llpidx[s] >= 0
            if not m.any():
                continue
            etas.append(eta[s][m])
            phis.append(phi[s][m])
            idxs.append(llpidx[s][m])
            if hit_dr is not None:
                srcs.append(np.flatnonzero(m) + offsets[i])
        if not etas:
            continue
        hit_eta = np.concatenate(etas)
        hit_phi = np.concatenate(phis)
        hit_idx = np.concatenate(idxs)
        src = np.concatenate(srcs) if hit_dr is not None else None

        # Group the matched hits by llpIdx with one sort, instead of one mask
        # per LLP (events hold up to O(50) LLPs, most without any rechit).
        order = np.argsort(hit_idx, kind="stable")
        idx_sorted = hit_idx[order]
        edges = np.flatnonzero(np.diff(idx_sorted)) + 1
        slot = {int(k): llp_off[i] + t
                for t, k in enumerate(keys_flat[llp_off[i]:llp_off[i + 1]])}

        for a, b in zip(np.concatenate([[0], edges]),
                        np.concatenate([edges, [len(order)]])):
            j = slot.get(int(idx_sorted[a]))
            if j is None:          # hits from a gen particle that is not an LLP
                continue
            sel = order[a:b]
            e, p = hit_eta[sel], hit_phi[sel]
            # Centroid: circular mean in phi, so hits either side of +-pi
            # average correctly (same convention as the DBSCAN clusters).
            cphi = np.arctan2(np.sin(p).sum(), np.cos(p).sum())
            dr = np.hypot(e - e.mean(), (p - cphi + np.pi) % (2 * np.pi) - np.pi)
            out["drMax"][j] = dr.max()
            for q, val in zip(DR_QUANTILES, np.quantile(dr, DR_QUANTILES)):
                out[f"dr{int(round(q * 100))}"][j] = val
            if hit_dr is not None:
                hit_dr[src[sel]] = dr
            if len(sel) >= 2:
                step = max(1, -(-len(sel) // PAIR_MAX_HITS))
                e2, p2 = e[::step], p[::step]
                dphi = p2[:, None] - p2[None, :]
                dphi = (dphi + np.pi) % (2 * np.pi) - np.pi
                dmat = np.hypot(e2[:, None] - e2[None, :], dphi)
                pair = dmat[np.triu_indices(len(e2), 1)]
                out["drPairMin"][j] = pair.min()
                out["drPairMax"][j] = pair.max()
    return out


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
              "nMatchedHits", "matchedLLPIdx", "matched", "hasTruth"]
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
            else:
                best, nbest = -1, 0
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
            out["hasTruth"].append(has_truth)
            if tvals is not None:
                out["time"].append(float(tvals[s][m].mean()))
        nclu[i] = labels.max() + 1

    dtypes = {"size": np.int64, "nStation": np.int64, "stationSpan": np.int64,
              "nLayer": np.int64, "nZLayer": np.int64, "nMatchedHits": np.int64,
              "matchedLLPIdx": np.int64, "matched": np.bool_,
              "hasTruth": np.bool_}
    return ak.zip({f: ak.unflatten(np.asarray(v, dtype=dtypes.get(f, np.float64)), nclu)
                   for f, v in out.items()})
