"""User-defined derived columns (coffea NanoEvents).

The processor calls ``derive(events)`` once per chunk, before filling
histograms, and uses the returned array in place of ``events``.  Add new fields
with :func:`awkward.with_field` and reference them in ``histograms.yaml`` /
``selections.yaml`` expressions as ``events.<name>``.

Return the (possibly augmented) ``events`` array.  If you don't need any derived
columns, the default pass-through below is fine.

MDS LLP cluster study (configs_mds/)
------------------------------------
For MDSNANO samples (detected by the presence of the ``cscRechits`` collection)
this module attaches four derived collections:

``events.llp``
    One entry per generated LLP (``SUEPGenPart.pdgId == 999999``): kinematics,
    decay vertex (from the production vertex of its daughters), decay-volume
    flags (``inCSC/inDT/inRPC``), per-system truth rechit counts
    (``nHitsCSC/nHitsDT/nHitsRPC/nHitsTotal``) and reconstruction flags
    (``recoCSC/recoDT/recoRPC/reco``: LLP has a matched DBSCAN cluster).

``events.cscCluster`` / ``events.dtCluster`` / ``events.rpcCluster``
    DBSCAN clusters of the muon-system rechits (per system, dR metric in
    eta-phi with proper phi wrap-around).  A cluster is truth-matched to an
    LLP when at least ``MATCH_MIN_HITS`` of its rechits carry that LLP's
    ``llpIdx``.

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
# size can be overridden with $MDS_CLUSTER_MIN_SAMPLES (default 50) to study
# its effect without editing the config; RPC stays at 10 (sparse system).
_CSC_DT_MIN_SAMPLES = int(os.environ.get("MDS_CLUSTER_MIN_SAMPLES", "50"))
DBSCAN_PARAMS = {
    "csc": (0.2, _CSC_DT_MIN_SAMPLES),
    "dt": (0.2, _CSC_DT_MIN_SAMPLES),
    "rpc": (0.2, 10),
}

# Cluster <-> LLP matching: matched := (# rechits sharing one llpIdx) >= MATCH_MIN_HITS
MATCH_MIN_HITS = 10


def derive(events):
    """Attach the MDS LLP/cluster collections (MDSNANO samples only)."""
    if "cscRechits" not in events.fields or "SUEPGenPart" not in events.fields:
        return events

    llp = _build_llps(events)

    clusters = {}
    for sys, coll, timefield in (("csc", "cscRechits", "Tpeak"),
                                 ("dt", "dtRecHits", None),
                                 ("rpc", "rpcRecHits", "Time")):
        eps, min_samples = DBSCAN_PARAMS[sys]
        clusters[sys] = _cluster_system(events[coll], timefield, eps, min_samples)

    # Per-LLP truth rechit counts and reconstruction flags
    nhits = {}
    for sys, coll in (("CSC", "cscRechits"), ("DT", "dtRecHits"), ("RPC", "rpcRecHits")):
        nhits[sys] = ak.values_astype(
            ak.sum(llp.gidx[:, :, None] == events[coll].llpIdx[:, None, :], axis=2),
            np.int64)
    llp = ak.with_field(llp, nhits["CSC"], "nHitsCSC")
    llp = ak.with_field(llp, nhits["DT"], "nHitsDT")
    llp = ak.with_field(llp, nhits["RPC"], "nHitsRPC")
    llp = ak.with_field(llp, nhits["CSC"] + nhits["DT"] + nhits["RPC"], "nHitsTotal")

    reco = {}
    for sys in ("csc", "dt", "rpc"):
        reco[sys] = ak.any(
            llp.gidx[:, :, None] == clusters[sys].matchedLLPIdx[:, None, :], axis=2)
    llp = ak.with_field(llp, reco["csc"], "recoCSC")
    llp = ak.with_field(llp, reco["dt"], "recoDT")
    llp = ak.with_field(llp, reco["rpc"], "recoRPC")
    llp = ak.with_field(llp, reco["csc"] | reco["dt"] | reco["rpc"], "reco")

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
    gidx = gp_idx[is_llp]  # SUEPGenPart index of each LLP == rechit llpIdx

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

    return ak.zip({
        "pt": pt,
        "eta": eta,
        "phi": gp.phi[is_llp],
        "mass": mass,
        "energy": np.hypot(pt * np.cosh(eta), mass),
        "gidx": gidx,
        "decayR": decay_r,
        "decayZ": dvz,
        "inCSC": in_csc,
        "inDT": in_dt,
        "inRPC": in_rpc,
    })


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
    llpidx = np.asarray(ak.flatten(rechits.llpIdx))
    tvals = (np.asarray(ak.flatten(rechits[timefield]), dtype=np.float64)
             if timefield is not None else None)

    fields = ["size", "eta", "phi", "x", "y", "z", "r", "etaSpread", "phiSpread",
              "nStation", "nMatchedHits", "matchedLLPIdx", "matched"]
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
            out["nStation"].append(len(np.unique(station[s][m])))
            out["nMatchedHits"].append(nbest)
            out["matchedLLPIdx"].append(best if matched else -1)
            out["matched"].append(matched)
            if tvals is not None:
                out["time"].append(float(tvals[s][m].mean()))
        nclu[i] = labels.max() + 1

    dtypes = {"size": np.int64, "nStation": np.int64, "nMatchedHits": np.int64,
              "matchedLLPIdx": np.int64, "matched": np.bool_}
    return ak.zip({f: ak.unflatten(np.asarray(v, dtype=dtypes.get(f, np.float64)), nclu)
                   for f, v in out.items()})
